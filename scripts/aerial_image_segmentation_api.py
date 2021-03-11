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
import math

import archs
from dataset import Dataset, DatasetPatch
from metrics import iou_score, dice_coef
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def post_process_resized_mask(resized_mask):
    half_th = 127
    mask_1 = ((resized_mask > half_th) & (resized_mask < 255))
    resized_mask[mask_1] = 255

    # tmp_maks = np.zeros((resized_mask.shape[0],resized_mask.shape[1]))
    mask_0 = ((resized_mask > 0) & (resized_mask <= half_th))
    resized_mask[mask_0] = 0

    return resized_mask


def patch_gen(img, mask, p_size, overlap=0.5):
    img_h = img.shape[0]
    img_w = img.shape[1]
    shift_size = 1 - overlap
    i_w = int(math.floor((img_w - p_size) / math.ceil(shift_size * p_size))) + 1
    i_h = int(math.floor((img_h - p_size) / math.ceil(shift_size * p_size))) + 1

    image_patch = []
    mask_patch = []

    h_step = int(math.ceil(shift_size * p_size))
    w_step = int(math.ceil(shift_size * p_size))
    for i in range(i_w):
        for j in range(i_h):
            idx_w1 = int(i * w_step)
            idx_w2 = idx_w1 + p_size
            idx_h1 = int(j * h_step)
            idx_h2 = int(idx_h1 + p_size)
            if idx_h1 < 0 or idx_w1 < 0:
                print('err')
            if idx_h2 > img_h or idx_w2 > img_w:
                print('err')
            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2, :])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2, :])

    for i in range(i_w):
        for j in range(i_h):
            idx_w2 = int(img_w - (i * w_step))
            idx_w1 = int(idx_w2 - p_size)
            idx_h2 = int(img_h - (j * h_step))
            idx_h1 = int(idx_h2 - p_size)
            if idx_h1 < 0 or idx_w1 < 0:
                print('err')
            if idx_h2 > img_h or idx_w2 > img_w:
                print('err')
            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2, :])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2, :])

    ## conner case

    for i in range(i_w):
        for j in range(i_h):
            idx_w1 = int((i * w_step))
            idx_w2 = int(idx_w1 + p_size)
            idx_h2 = int(img_h - (j * h_step))
            idx_h1 = int(idx_h2 - p_size)
            if (idx_h2 - idx_h1) != p_size or (idx_w2 - idx_w1) != p_size:
                print('err')
            if idx_h1 < 0 or idx_w1 < 0:
                print('err')
            if idx_h2 > img_h or idx_w2 > img_w:
                print('err')
            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2, :])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2, :])

    for i in range(i_w):
        for j in range(i_h):
            idx_w2 = int(img_w - (i * w_step))
            idx_w1 = int(idx_w2 - p_size)
            idx_h1 = int(j * h_step)
            idx_h2 = int(idx_h1 + p_size)
            if (idx_h2 - idx_h1) != p_size or (idx_w2 - idx_w1) != p_size:
                print('err')
            if idx_h1 < 0 or idx_w1 < 0:
                print('err')
            if idx_h2 > img_h or idx_w2 > img_w:
                print('err')

            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2, :])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2, :])

    return image_patch, mask_patch


def patch_merge(img, masks, p_size, config, p_overlap):
    img_h = img.shape[0]
    img_w = img.shape[1]
    shift_size = 1 - p_overlap
    i_w = int(math.floor((img_w - p_size) / math.ceil(shift_size * p_size))) + 1
    i_h = int(math.floor((img_h - p_size) / math.ceil(shift_size * p_size))) + 1

    mask_o = np.ones((p_size, p_size))
    h_step = int(math.ceil(shift_size * p_size))
    w_step = int(math.ceil(shift_size * p_size))

    all_class_mask = []
    for c in range(config['num_classes']):
        merged_mask = np.zeros((img_h, img_w))
        mask_merge_div = np.zeros((img_h, img_w))
        p_idx = 0
        for i in range(i_w):
            for j in range(i_h):
                idx_w1 = int(i * w_step)
                idx_w2 = idx_w1 + p_size
                idx_h1 = int(j * h_step)
                idx_h2 = idx_h1 + p_size
                mask = masks[p_idx][c]
                p_idx += 1
                mask = (mask * 255).astype('uint8')
                resized_mask = cv2.resize(mask, (p_size, p_size))
                resized_mask = post_process_resized_mask(resized_mask) / 255.0
                merged_mask[idx_h1:idx_h2, idx_w1:idx_w2] += resized_mask
                mask_merge_div[idx_h1:idx_h2, idx_w1:idx_w2] += mask_o

        for i in range(i_w):
            for j in range(i_h):
                idx_w2 = int(img_w - (i * w_step))
                idx_w1 = int(idx_w2 - p_size)

                idx_h2 = int(img_h - (j * h_step))
                idx_h1 = idx_h2 - p_size
                mask = masks[p_idx][c]
                p_idx += 1
                mask = (mask * 255).astype('uint8')
                resized_mask = cv2.resize(mask, (p_size, p_size))
                resized_mask = post_process_resized_mask(resized_mask) / 255.0
                merged_mask[idx_h1:idx_h2, idx_w1:idx_w2] += resized_mask
                mask_merge_div[idx_h1:idx_h2, idx_w1:idx_w2] += mask_o

        ## conner case

        for i in range(i_w):
            for j in range(i_h):
                idx_w1 = int((i * w_step))
                idx_w2 = int(idx_w1 + p_size)
                idx_h2 = int(img_h - (j * h_step))
                idx_h1 = int(idx_h2 - p_size)

                if (idx_h2 - idx_h1) != p_size or (idx_w2 - idx_w1) != p_size:
                    print('err')
                if idx_h1 < 0 or idx_w1 < 0:
                    print('err')
                if idx_h2 > img_h or idx_w2 > img_w:
                    print('err')

                mask = masks[p_idx][c]
                p_idx += 1
                mask = (mask * 255).astype('uint8')
                resized_mask = cv2.resize(mask, (p_size, p_size))
                resized_mask = post_process_resized_mask(resized_mask) / 255.0
                merged_mask[idx_h1:idx_h2, idx_w1:idx_w2] += resized_mask
                mask_merge_div[idx_h1:idx_h2, idx_w1:idx_w2] += mask_o

        for i in range(i_w):
            for j in range(i_h):
                idx_w2 = int(img_w - (i * w_step))
                idx_w1 = int(idx_w2 - p_size)
                idx_h1 = int(j * h_step)
                idx_h2 = int(idx_h1 + p_size)

                if (idx_h2 - idx_h1) != p_size or (idx_w2 - idx_w1) != p_size:
                    print('err')
                if idx_h1 < 0 or idx_w1 < 0:
                    print('err')
                if idx_h2 > img_h or idx_w2 > img_w:
                    print('err')

                mask = masks[p_idx][c]
                p_idx += 1
                mask = (mask * 255).astype('uint8')
                resized_mask = cv2.resize(mask, (p_size, p_size))
                resized_mask = post_process_resized_mask(resized_mask) / 255.0
                merged_mask[idx_h1:idx_h2, idx_w1:idx_w2] += resized_mask
                mask_merge_div[idx_h1:idx_h2, idx_w1:idx_w2] += mask_o

        mask_merge_div.astype('float32')
        _zero_idx = (mask_merge_div == 0)
        mask_merge_div[_zero_idx] = 1.0
        full_mask = np.divide(merged_mask, mask_merge_div)
        full_mask = (full_mask * 255).astype('uint8')
        full_mask = post_process_resized_mask(full_mask)
        all_class_mask.append(full_mask)
    return all_class_mask


def mask_convert(p_mask, idx, p_size):
    mask = np.zeros((p_mask.shape[0], p_mask.shape[1]))
    if idx == 0:
        mask_ = ((p_mask[:, :, 0] == 255) & (p_mask[:, :, 1] == 255) & (p_mask[:, :, 2] == 255))
    if idx == 1:
        mask_ = ((p_mask[:, :, 0] == 255) & (p_mask[:, :, 1] == 0) & (p_mask[:, :, 2] == 0))

    if idx == 2:
        mask_ = ((p_mask[:, :, 0] == 0) & (p_mask[:, :, 1] == 0) & (p_mask[:, :, 2] == 255))

    mask[mask_] = 1
    mask = (mask * 255).astype('uint8')

    resized_mask = cv2.resize(mask, (p_size, p_size))
    resized_mask = post_process_resized_mask(resized_mask)

    return resized_mask


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

    cv2.imwrite(save_name, img)


def save_masking(img, mask_GT, mask_out, save_name):
    yellow = np.array([0, 255, 255]).astype('uint8')
    pink = np.array([255, 0, 255]).astype('uint8')

    idx = mask_GT > 0
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


def load_segmentation_models(config_file):
    config_dict = json.loads(open(config_file, 'rt').read())
    file_dict = config_dict['file_path']
    val_config = config_dict['val_config']
    model_folder = file_dict['model_path']
    name = val_config['name']

    with open(os.path.join(model_folder, '%s/config.yml' % name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['name'] = name
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(model_folder, '%s/model.pth' % config['name'])))
    model.eval()

    config['patch_size'] = 1024
    config['patch_overlap'] = val_config['patch_overlap']

    return model, config


def get_patched_input(img_path, config, gt_mask_flag):
    img_patch_set = []
    p_size = config['patch_size']
    img_size = config['input_w']
    patch_overlap = config['patch_overlap']

    if gt_mask_flag == True:
        label_path = img_path.replace('image', 'labels')

    img_input = cv2.imread(img_path)
    if gt_mask_flag == True:
        mask_input = cv2.imread(label_path)

    if gt_mask_flag == True:
        image_patch, mask_patch = patch_gen(img_input, mask_input, p_size, patch_overlap)
    else:
        image_patch, mask_patch = patch_gen(img_input, img_input, p_size, patch_overlap)

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    patch_len = len(image_patch)
    for idx in range(patch_len):
        img = image_patch[idx]
        img = cv2.resize(img, (img_size, img_size))
        mask = img
        if val_transform is not None:
            augmented = val_transform(image=img, mask=mask)
            img = augmented['image']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img_patch_set.append(img)
    img_patch_set = np.array(img_patch_set)
    mask_patch_set = np.array(mask_patch)

    return img_input, img_patch_set, mask_patch_set


def segmentation_inference(model, img_input, img_patch_set, mask_patch_set, config, gt_mask_flag):
    patch_size = config['patch_size']
    infer_size = config['input_w']
    p_overlap = config['patch_overlap']

    input = torch.from_numpy(img_patch_set)
    input = input.cuda()

    # compute output
    full_output = []
    for idx, data in enumerate(input):
        in_tmp = torch.unsqueeze(data, 0)
        output = model(in_tmp)
        output = torch.sigmoid(output).cpu().detach().numpy()
        full_output.append(np.squeeze(output, 0))
    full_img = img_input
    all_class_mask = patch_merge(full_img, full_output, patch_size, config, p_overlap)

    if gt_mask_flag == True:
        target = torch.from_numpy(mask_patch_set)
        gt_label = []
        for idx, data in enumerate(target):
            mask_tmp = []
            for c in range(config['num_classes']):
                con_mask = mask_convert(data, c, infer_size)
                mask_tmp.append(con_mask)
            mask_patch = np.dstack(mask_tmp)
            mask_patch = mask_patch.transpose(2, 0, 1)
            gt_label.append(mask_patch / 255.0)

        gt_class_mask = patch_merge(full_img, gt_label, patch_size, config, p_overlap)
    else:
        gt_class_mask = all_class_mask

    return all_class_mask, gt_class_mask


def save_image_color_masking(output_folder, image_name, full_img, all_class_mask, gt_class_mask, config, gt_mask_flag):
    for c in range(config['num_classes']):
        file_name = '{:s}_{:d}'.format(image_name, c)
        full_img = np.array(full_img)
        if c > 0:
            if gt_mask_flag == True:
                save_name = os.path.join(output_folder, config['name'], file_name + '_GT_RE_masking.jpg')
                save_name_GT = os.path.join(output_folder, config['name'], file_name + '_GT_masking.jpg')
            save_name_RE = os.path.join(output_folder, config['name'], file_name + '_RE_masking.jpg')
            mask_output = all_class_mask[c]
            if gt_mask_flag == True:
                mask_gt = gt_class_mask[c]
            # save_contour(np.array(full_img), mask_gt, mask_output, save_name)
            if gt_mask_flag == True:
                save_img = full_img.copy()
                save_masking_GT(np.array(save_img), mask_gt, mask_output, save_name_GT)
            save_img = full_img.copy()
            save_masking_RE(np.array(save_img), mask_output, mask_output, save_name_RE)
            if gt_mask_flag == True:
                save_img = full_img.copy()
                save_masking(np.array(save_img), mask_gt, mask_output, save_name)

    return 0


