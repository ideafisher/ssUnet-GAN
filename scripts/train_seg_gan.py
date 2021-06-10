import argparse
import os
from collections import OrderedDict
from glob import glob
import sys
import cv2
import numpy as np
import json
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchsummary import summary
import random
import archs
import losses
from losses import masked_L1_loss
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter, str2bool
from models_seg_gan import Generator, Discriminator, TruncatedVGG19
from srgan_utils import *

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

torch_seed = 41
torch.manual_seed(torch_seed)
random_seed = 101
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deerministic = True
torch.backends.cudnn.benchmark = True


def save_tensorboard(writer, train_log, val_log, test_log, epoch):
    writer.add_scalar('train loss per epoch', train_log['loss'], epoch)
    writer.add_scalar('train iou per epoch', train_log['iou'], epoch)
    writer.add_scalar('train dice per epoch', train_log['dice'], epoch)
    writer.add_scalar('val loss per epoch', val_log['loss'], epoch)
    writer.add_scalar('val iou per epoch', val_log['iou'], epoch)
    writer.add_scalar('val dice per epoch', val_log['dice'], epoch)
    writer.add_scalar('test loss per epoch', test_log['loss'], epoch)
    writer.add_scalar('test iou per epoch', test_log['iou'], epoch)
    writer.add_scalar('test dice per epoch', test_log['dice'], epoch)

def optimizer_scheduler(params, config):
    if config['optimizer'] == 'Adam':
        optimizer_g = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer_g = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                                nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['optimizer'] == 'SGD':
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer_g, T_max=config['epochs'], eta_min=config['min_lr'])
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_g, factor=config['factor'],
                                                       patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer_g,
                                                 milestones=[int(e) for e in config['milestones'].split(',')],
                                                 gamma=config['gamma'])
        elif config['scheduler'] == 'ConstantLR':
            scheduler = None
        else:
            raise NotImplementedError
    else:
        scheduler = None

    return optimizer_g, scheduler

def parse_args_func():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--log_name', default='train_results', type=str)

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--aug_type', type=str, default='medical_mode')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def remove_prefix(state_dict, prefix):
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def train(epoch, config, train_loader,   generator,
              discriminator, criterion, adversarial_loss_criterion, content_loss_criterion,  optimizer_g, optimizer_d):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    alpa = 1e-4
    beta = 1e-3
    grad_clip = 0.8
    generator.train()
    discriminator.train()
    lr_val = optimizer_g.param_groups[0]['lr']
    print('generator learning rate {:d}: {:f}'.format(epoch, lr_val))
    pbar = tqdm(total=len(train_loader))
    num_class = int(config['num_classes'])

    for ori_img, input, target, targets, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        # input[torch.isnan(input)] = 0
        generator_output = generator(input)  # (N, 3, 96, 96), in [-1, 1]
        # l1_loss = masked_L1_loss(input, target, output)
        generator_output[torch.isnan(generator_output)] = 0
        out_m = generator_output[:, 1:num_class, :, :].clone()
        tar_m = target[:, 1:num_class, :, :].clone()

        loss = criterion(generator_output, target)
        content_loss = content_loss_criterion(generator_output, target)
        # loss = criterion(out_m, tar_m)
        iou = iou_score(out_m, tar_m)
        dice = dice_coef(out_m, tar_m)
        # iou = iou_score(output, target)
        # dice = dice_coef(output, target)

        seg_discriminated = discriminator(generator_output)  # (N)

        adversarial_loss = adversarial_loss_criterion(seg_discriminated, torch.ones_like(seg_discriminated))
        perceptual_loss = loss + alpa * content_loss + beta * adversarial_loss
        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        hr_discriminated = discriminator(target)
        sr_discriminated = discriminator(generator_output.detach())

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, generator, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    generator.eval()
    num_class = int(config['num_classes'])
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for ori_img, input, target, targets, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = generator(input)
            output[torch.isnan(output)] = 0
            out_m = output[:, 1:num_class, :, :].clone()
            tar_m = target[:, 1:num_class, :, :].clone()
            loss = criterion(output, target)
            # loss = criterion(out_m, tar_m)
            iou = iou_score(out_m, tar_m)
            dice = dice_coef(out_m, tar_m)
            # iou = iou_score(output, target)
            # dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    # config = vars(parse_args_func())

    #config_file = "../configs/config_v1.json"
    args = vars(parse_args_func())
    config_file = args['config']
    config_dict = json.loads(open(config_file, 'rt').read())
    # config_dict = json.loads(open(sys.argv[1], 'rt').read())

    file_dict = config_dict['file_path']
    config = config_dict['opt_config']

    input_folder = file_dict['input_path']  # '../inputs'
    checkpoint_folder = file_dict['checkpoint_path']  # '../checkpoint'
    model_folder = file_dict['model_path']  # '../models'

    if 'False' in config['deep_supervision']:
        config['deep_supervision'] = False
    else:
        config['deep_supervision'] = True

    if 'False' in config['nesterov']:
        config['nesterov'] = False
    else:
        config['nesterov'] = True

    if 'None' in config['name']:
        config['name'] = None

    if config['name'] is None:
        config['name'] = '%s_%s_segmodel' % (config['dataset'], config['arch'])
    os.makedirs(os.path.join(model_folder, '%s' % config['name']), exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(os.path.join(model_folder, '%s/config.yml' % config['name']), 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model

    if 'False' in config['resume']:
        config['resume'] = False
    else:
        config['resume'] = True

    # Data loading code
    img_ids = glob(os.path.join(input_folder, config['dataset'], 'images', 'training', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    img_ids = glob(os.path.join(input_folder, config['val_dataset'], 'images', 'validation', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    img_ids = glob(os.path.join(input_folder, config['val_dataset'], 'images', 'test', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = Compose([
        # transforms.RandomScale ([config['scale_min'], config['scale_max']]),
        # transforms.RandomRotate90(),
        transforms.Rotate([config['rotate_min'], config['rotate_max']], value=mean, mask_value=0),
        # transforms.GaussianBlur (),
        transforms.Flip(),
        # transforms.HorizontalFlip (),
        transforms.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
        transforms.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, brightness_by_max=True),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(input_folder, config['dataset'], 'images', 'training'),
        mask_dir=os.path.join(input_folder, config['dataset'], 'annotations', 'training'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(input_folder, config['dataset'], 'images', 'validation'),
        mask_dir=os.path.join(input_folder, config['dataset'], 'annotations', 'validation'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        transform=val_transform)
    test_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(input_folder, config['dataset'], 'images', 'test'),
        mask_dir=os.path.join(input_folder, config['dataset'], 'annotations', 'test'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    # create generator model
    #val_config = config_dict['config']
    generator_name = config['generator_name']
    with open(os.path.join(model_folder,'%s/config.yml' % generator_name), 'r') as f:
        g_config = yaml.load(f, Loader=yaml.FullLoader)
    generator = Generator(g_config)
    generator.initialize_with_srresnet(model_folder, g_config)
    lr = config['gan_lr']
    # Initialize generator's optimizer
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
    #params = filter(lambda p: p.requires_grad, generator.parameters())
    #optimizer_g, scheduler_g = optimizer_scheduler(params, config)

    # Discriminator
    # Discriminator parameters
    num_classes = config['num_classes']
    kernel_size_d = 3  # kernel size in all convolutional blocks
    n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
    n_blocks_d = 8  # number of convolutional blocks
    fc_size_d = 1024  # size of the first fully connected layer
    discriminator = Discriminator(num_classes, kernel_size=kernel_size_d,
                                  n_channels=n_channels_d,
                                  n_blocks=n_blocks_d,
                                  fc_size=fc_size_d)
    # Initialize discriminator's optimizer
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),lr=lr)
    #params = filter(lambda p: p.requires_grad, discriminator.parameters())
    #optimizer_d, scheduler_d = optimizer_scheduler(params, config)
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()
    content_loss_criterion = nn.MSELoss()

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    #truncated_vgg19 = truncated_vgg19.cuda()
    content_loss_criterion = content_loss_criterion.cuda()
    adversarial_loss_criterion = adversarial_loss_criterion.cuda()

    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)


    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    log_name = config['name']
    log_dir = os.path.join(checkpoint_folder, log_name)
    writer = SummaryWriter(logdir=log_dir)

    best_iou = 0
    trigger = 0
    Best_dice = 0
    iou_AtBestDice = 0
    start_epoch = 0
    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(epoch, config, train_loader,  generator,
              discriminator, criterion, adversarial_loss_criterion, content_loss_criterion, optimizer_g, optimizer_d)

        # evaluate on validation set
        val_log = validate(config, val_loader, generator, criterion)
        test_log = validate(config, test_loader, generator, criterion)


        if Best_dice < test_log['dice']:
            Best_dice = test_log['dice']
            iou_AtBestDice = test_log['iou']
        print(
            'loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - test_iou %.4f - test_dice %.4f - Best_dice %.4f - iou_AtBestDice %.4f'
            % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice'],
               test_log['iou'], test_log['dice'], Best_dice, iou_AtBestDice))

        save_tensorboard(writer, train_log, val_log, test_log, epoch)
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(os.path.join(model_folder, '%s/log.csv' % config['name']), index=False)
        trigger += 1

        if test_log['iou'] > best_iou:
            torch.save(generator.state_dict(), os.path.join(model_folder, '%s/model.pth' % config['name']))
            best_iou = test_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
