import argparse
import os
from glob import glob
import json, sys
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict
import archs
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
from models_seg_gan import Generator, Discriminator, TruncatedVGG19
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args

def result_save_to_csv_filename(csv_save_name, result_submission):

    result_submission = pd.DataFrame(result_submission, columns=['filename', 'iou', 'dice'])
    result_submission.sort_values('filename').to_csv(csv_save_name, index=False)

def save_contour(img, mask_GT, mask_out, save_name):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(mask_GT, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (255, 0, 0), 2)


    ret, img_binary = cv2.threshold(mask_out, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    #result_dir = './checkpoint/result'
    #name = os.path.join(result_dir, 'result_dir_{:d}.png'.format(2))
    cv2.imwrite(save_name, img)

def save_masking_GT_RE(img, mask_GT, mask_out, save_name):
    yellow = np.array([0,255,255]).astype('uint8')
    pink = np.array([255, 0, 255]).astype('uint8')
    idx = mask_GT >0
    img[idx, 0] = 0.5 * yellow[0] + 0.5 * img[idx, 0]
    img[idx, 1] = 0.5 * yellow[1] + 0.5 * img[idx, 1]
    img[idx, 2] = 0.5 * yellow[2] + 0.5 * img[idx, 2]

    idx = mask_out > 0
    img[idx, 0] = 0.5 * pink[0] + 0.5 * img[idx, 0]
    img[idx, 1] = 0.5 * pink[1] + 0.5 * img[idx, 1]
    img[idx, 2] = 0.5 * pink[2] + 0.5 * img[idx, 2]
    img.astype('uint8')

    cv2.imwrite(save_name, img)
    return 0


def save_masking_RE(img, mask_GT, mask_out, save_name):
    yellow = np.array([0, 255, 255]).astype('uint8')
    pink = np.array([255, 0, 255]).astype('uint8')

    idx = mask_out > 0
    img[idx, 0] = 0.5 * pink[0] + 0.5 * img[idx, 0]
    img[idx, 1] = 0.5 * pink[1] + 0.5 * img[idx, 1]
    img[idx, 2] = 0.5 * pink[2] + 0.5 * img[idx, 2]
    img.astype('uint8')

    cv2.imwrite(save_name, img)
    return 0


def save_masking_GT(img, mask_GT, mask_out, save_name):
    yellow = np.array([0, 255, 255]).astype('uint8')
    pink = np.array([255, 0, 255]).astype('uint8')

    idx = mask_GT > 0
    img[idx, 0] = 0.5 * yellow[0] + 0.5 * img[idx, 0]
    img[idx, 1] = 0.5 * yellow[1] + 0.5 * img[idx, 1]
    img[idx, 2] = 0.5 * yellow[2] + 0.5 * img[idx, 2]
    img.astype('uint8')

    cv2.imwrite(save_name, img)
    return 0

def remove_prefix(state_dict, prefix):
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def main():
    args = parse_args()
    config_file = "../configs/config_v1.json"
    config_dict = json.loads(open(config_file, 'rt').read())
    #config_dict = json.loads(open(sys.argv[1], 'rt').read())

    file_dict = config_dict['file_path']
    val_config = config_dict['val_config']

    name = val_config['name']
    input_folder  =file_dict['input_path'] # '../inputs'
    model_folder = file_dict['model_path']  # '../models'
    output_folder = file_dict['output_path']  # '../models'

    ss_unet_GAN = True
    # create model
    if ss_unet_GAN == False:
        with open(os.path.join(model_folder, '%s/config.yml' % name), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config['name'] = name
        print('-' * 20)
        for key in config.keys():
            print('%s: %s' % (key, str(config[key])))
        print('-' * 20)
        cudnn.benchmark = True
        print("=> creating model %s" % config['arch'])
        model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
        model = model.cuda()

        #img_ids = glob(os.path.join(input_folder, config['dataset'], 'images', '*' + config['img_ext']))
        #img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        #_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        model_dict = torch.load(os.path.join(model_folder,'%s/model.pth' %config['name']))
        if "state_dict" in model_dict.keys():
            model_dict = remove_prefix(model_dict['state_dict'], 'module.')
        else:
            model_dict = remove_prefix(model_dict, 'module.')
        model.load_state_dict(model_dict, strict=False)
        #model.load_state_dict(torch.load(os.path.join(model_folder,'%s/model.pth' %config['name'])))
        model.eval()
    else:
        val_config = config_dict['val_config']
        generator_name = val_config['name']
        with open(os.path.join(model_folder, '%s/config.yml' % generator_name), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        generator = Generator(config)
        generator = generator.cuda()
        '''
        with open(os.path.join(model_folder, '%s/config.yml' % name), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        '''
        config['name'] = name
        model_dict = torch.load(os.path.join(model_folder,'%s/model.pth' %config['name']))
        if "state_dict" in model_dict.keys():
            model_dict = remove_prefix(model_dict['state_dict'], 'module.')
        else:
            model_dict = remove_prefix(model_dict, 'module.')
        generator.load_state_dict(model_dict, strict=False)
        #model.load_state_dict(torch.load(os.path.join(model_folder,'%s/model.pth' %config['name'])))
        generator.eval()

    # Data loading code
    img_ids = glob(os.path.join(input_folder, config['val_dataset'], 'images','test', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        transforms.Normalize(mean=mean, std=std),
    ])


    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(input_folder, config['val_dataset'], 'images','test'),
        mask_dir=os.path.join(input_folder, config['val_dataset'], 'annotations','test'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, #config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meters = {'iou': AverageMeter(),
                  'dice' : AverageMeter()}

    num_classes = config['num_classes']
    for c in range(config['num_classes']):
        os.makedirs(os.path.join( output_folder, config['name'], str(c)), exist_ok=True)

    csv_save_name = os.path.join(output_folder, config['name'] + '_result' + '.csv')
    result_submission = []
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for ori_img, input, target, targets,  meta in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if ss_unet_GAN == True:
                if config['deep_supervision']:
                    output = generator(input)[-1]
                else:
                    output = generator(input)
            else:
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)
            out_m = output[:, 1:num_classes, :, :].clone()
            tar_m = target[:, 1:num_classes, :, :].clone()
            iou = iou_score(out_m, tar_m)
            dice = dice_coef(out_m, tar_m)
            result_submission.append([meta['img_id'][0], iou, dice])
            #iou = iou_score(output, target)
            #dice = dice_coef(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            imgs = input.cpu()
            masks = target.cpu()
            for i in range(len(output)):

                if num_classes <1 :
                    img = np.array(ori_img[i])
                    mask = np.array(255*masks[i]).astype('uint8').squeeze(0)
                    mask_out = np.array(255 * output[i]).astype('uint8').squeeze(0)

                    save_name = os.path.join(output_folder, config['name']+'_contour_output', meta['img_id'][i] + '.jpg')
                    save_contour(img, mask, mask_out, save_name)
                    for c in range(config['num_classes']):
                        cv2.imwrite(os.path.join(output_folder, config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                    (output[i, c] * 255).astype('uint8'))
                else:

                    for idx_c in range(num_classes):
                        img = np.array(ori_img[i])
                        tmp_mask = np.array(masks[i][idx_c])
                        mask = np.array(255 * tmp_mask).astype('uint8')
                        mask_out = np.array(255 * output[i][idx_c]).astype('uint8')

                        mask_output = np.zeros((mask_out.shape[0], mask_out.shape[1]))
                        mask_output = mask_output.astype('uint8')
                        mask_ = mask_out > 127
                        mask_output[mask_] = 255

                        #save_name = os.path.join('outputs', config['name'], str(idx_c),  meta['img_id'][i] + '.jpg')
                        #save_contour(img, mask, mask_output, save_name)

                        if idx_c >0:
                            save_name_GT = os.path.join(output_folder, config['name'], str(idx_c), meta['img_id'][i]+' _GT_masking' + '.jpg')
                            save_name_RE = os.path.join(output_folder, config['name'], str(idx_c), meta['img_id'][i] + '_RE_masking' + '.jpg')
                            save_name_GT_RE = os.path.join(output_folder, config['name'], str(idx_c), meta['img_id'][i] + '_GT_RE_masking' + '.jpg')
                            img = np.array(ori_img[i])
                            save_masking_GT(img, mask, mask_output, save_name_GT)
                            img = np.array(ori_img[i])
                            save_masking_RE(img, mask, mask_output, save_name_RE)
                            img = np.array(ori_img[i])
                            save_masking_GT_RE(img, mask, mask_output, save_name_GT_RE)
                        #cv2.imwrite(os.path.join('outputs', config['name'], str(idx_c), meta['img_id'][i]+'_mask_' + '.jpg'), mask_output)

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    result_save_to_csv_filename(csv_save_name, result_submission)
    print('IoU: %.4f' % avg_meters['iou'].avg)
    print('dice: %.4f' % avg_meters['dice'].avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
