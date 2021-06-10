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
import torchvision.transforms as tv_transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchsummary import summary
import random
import archs
import losses
from losses import masked_L1_loss
from dataset import Dataset, image_to_afile
from metrics import iou_score, dice_coef
from utils import AverageMeter, str2bool

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


def parse_args_func():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--config', type=str,default='../configs/config_v1.json',
                        help='config file')

    config = parser.parse_args()

    return config



def train(epoch, config, train_loader, model, criterion, optimizer, cnn_optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()
    clip =  float(config['clip'])
    lr_val = optimizer.param_groups[0]['lr']
    print('learning rate {:d}: {:f}'.format(epoch, lr_val))
    pbar = tqdm(total=len(train_loader))
    num_class = int(config['num_classes'])
    for ori_img, input, target, targets, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            #avg_output = torch.zeros(outputs[-1].shape)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
                #avg_output += output.cpu()
            loss /= len(outputs)
            #avg_output /= len(outputs)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(outputs[-1], target)
            #dice = dice_coef(avg_output, target)
        else:
            #input[torch.isnan(input)] = 0
            output = model(input)
            #l1_loss = masked_L1_loss(input, target, output)
            output[torch.isnan(output)] = 0
            out_m = output[:, 1:num_class, :, :].clone()
            tar_m = target[:, 1:num_class, :, :].clone()

            loss = criterion(output, target)
            #loss = criterion(out_m, tar_m)
            iou = iou_score(out_m, tar_m)
            dice = dice_coef(out_m, tar_m)
            #iou = iou_score(output, target)
            #dice = dice_coef(output, target)

        for p in model.parameters():
            p.data.clamp_(-clip, clip)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Start CNN fine-tuning
        if cnn_optimizer != None:
            if epoch > 1:
                cnn_optimizer.step()

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



def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice' : AverageMeter()}

    # switch to evaluate mode
    model.eval()
    num_class = int(config['num_classes'])
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for ori_img, input, target, targets,  _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
            else:
                #input[torch.isnan(input)] = 0
                output = model(input)
                output[torch.isnan(output)] = 0
                out_m = output[:, 1:num_class, :, :].clone()
                tar_m = target[:, 1:num_class, :, :].clone()
                loss = criterion(output, target)
                #loss = criterion(out_m, tar_m)
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
    args = vars(parse_args_func())

    #config_file = "../configs/config_SN7.json"
    config_file = args['config'] # "../configs/config_v1.json"
    config_dict = json.loads(open(config_file, 'rt').read())
    #config_dict = json.loads(open(sys.argv[1], 'rt').read())

    file_dict = config_dict['file_path']
    config = config_dict['opt_config']

    input_folder  =file_dict['input_path'] # '../inputs'
    checkpoint_folder = file_dict['checkpoint_path'] # '../checkpoint'
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
    os.makedirs(os.path.join(model_folder,'%s' % config['name']), exist_ok=True)

    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    log_name = config['name']
    log_dir = os.path.join(checkpoint_folder,log_name )
    writer = SummaryWriter(logdir=log_dir)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(os.path.join(model_folder,'%s/config.yml' % config['name']), 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    if 'False' in config['resume']:
        config['resume'] = False
    else:
        config['resume'] = True
    resume_flag = False
    if resume_flag == True:
        save_path = os.path.join(model_folder,config['name'],'model.pth')
        weights = torch.load(save_path)
        model.load_state_dict(weights)
        name_yaml = config['name']
        with open(os.path.join(model_folder ,'%s/config.yml' %name_yaml), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        #start_epoch = config['epochs']
        start_epoch = 0
    else:
        start_epoch = 0

    model = model.cuda()
    if 'effnet' in config['arch']:
        eff_flag = True
    else:
        eff_flag = False

    if eff_flag == True:
        cnn_subs = list(model.encoder.eff_conv.children())[1:]
        #cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
        #cnn_params = [item for sublist in cnn_params for item in sublist]

    summary(model, (config['input_channels'], config['input_w'], config['input_h']))
    params = filter(lambda p: p.requires_grad, model.parameters())
    if eff_flag == True:
        params = list(params) + list(model.encoder.conv_a.parameters())
    model = torch.nn.DataParallel(model)

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam( params, lr=config['lr'],weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if eff_flag == True:
        cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
        cnn_params = [item for sublist in cnn_params for item in sublist]
        cnn_optimizer = torch.optim.Adam(cnn_params, lr=0.001, weight_decay=config['weight_decay'])
        #cnn_optimizer = None

    else:
        cnn_optimizer = None
    if  config['optimizer'] == 'SGD':
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
        elif config['scheduler'] == 'ConstantLR':
            scheduler = None
        else:
            raise NotImplementedError
    else:
        scheduler = None

    # Data loading code
    img_ids = glob(os.path.join(input_folder, config['dataset'], 'images','training', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #img_dir = os.path.join(input_folder, config['dataset'], 'images', 'training')
    #mask_dir = os.path.join(input_folder, config['dataset'], 'annotations', 'training')
    #train_image_mask = image_to_afile(img_dir, mask_dir, None, train_img_ids, config)

    img_ids = glob(os.path.join(input_folder, config['val_dataset'], 'images','validation', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    img_ids = glob(os.path.join(input_folder, config['val_dataset'], 'images','test', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = Compose([
        #transforms.RandomScale ([config['scale_min'], config['scale_max']]),
        #transforms.RandomRotate90(),
        transforms.Rotate([config['rotate_min'],  config['rotate_max']], value=mean, mask_value = 0),
        transforms.Flip(),
        #transforms.HorizontalFlip (),
        transforms.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
        transforms.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, brightness_by_max=True),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(mean=mean,std=std),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(input_folder, config['dataset'], 'images', 'training'),
        mask_dir=os.path.join(input_folder, config['dataset'], 'annotations','training'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels = config['input_channels'],
        transform=train_transform,
        from_file=None)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(input_folder, config['val_dataset'], 'images','validation'),
        mask_dir=os.path.join(input_folder, config['val_dataset'], 'annotations','validation'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        transform=val_transform,
        from_file=None)
    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(input_folder, config['val_dataset'], 'images','test'),
        mask_dir=os.path.join(input_folder, config['val_dataset'], 'annotations','test'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        transform=val_transform,
        from_file=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, #config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, #config['batch_size'],
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


    best_iou = 0
    trigger = 0
    Best_dice = 0
    iou_AtBestDice = 0
    for epoch in range(start_epoch, config['epochs']):
        print('{:s} Epoch [{:d}/{:d}]'.format(config['arch'], epoch, config['epochs']))
        # train for one epoch
        train_log = train(epoch, config, train_loader, model, criterion, optimizer, cnn_optimizer)
        if config['optimizer'] == 'SGD':
            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            elif config['scheduler'] == 'MultiStepLR':
                scheduler.step()

        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        test_log = validate(config, test_loader, model, criterion)

        if Best_dice <  test_log['dice']:
            Best_dice = test_log['dice']
            iou_AtBestDice = test_log['iou']
        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - test_iou %.4f - test_dice %.4f - Best_dice %.4f - iou_AtBestDice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice'],  test_log['iou'], test_log['dice'],Best_dice, iou_AtBestDice))

        save_tensorboard(writer, train_log, val_log, test_log, epoch)
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(os.path.join(model_folder,'%s/log.csv' % config['name']), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(model_folder,'%s/model.pth' % config['name']))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
