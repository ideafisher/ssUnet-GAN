import os

import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob
import math

def save_contour(img, mask):

    d_size = len(mask.shape)
    if d_size == 3:
        mask = np.squeeze(mask, 0)
    ret, img_binary = cv2.threshold(mask, 127, 255, 0)
    val = np.max(img_binary)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (255, 0, 0), 3)

    result_dir = './checkpoint/result'
    name = os.path.join(result_dir, 'result_dir_{:d}.png'.format(4))
    cv2.imwrite(name, img)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_channels, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        if self.input_channels == 3:
            img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        else:
            img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext), cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img,0)
            img = img.transpose(1, 2, 0)
        ori_img = img
        if self.num_classes > 1 :
            is_single_mask = False
        else:
            is_single_mask = True

        if is_single_mask == True:
            path = os.path.join(self.mask_dir, img_id + self.mask_ext)
            mask =   cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)

            #save_contour(img, mask)
            #mask = np.dstack(mask)
            mask = np.expand_dims(mask,0)
            #save_contour(img, mask)
            mask = mask.transpose(1, 2, 0)
            mask = (mask / 1.0).astype('uint8')
            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            img_size = img.shape
            img_w, img_h = img_size[0], img_size[1]

            masks = []
            if 0:
                mask_one = mask.transpose(2, 0, 1)
                mask_one = mask_one.astype('float32') / 255
                masks.append(mask_one)
                for idx in range(3):
                    img_w = int(img_w/2)
                    img_h = int(img_h/2)
                    mask1 = cv2.resize(mask, (img_w, img_h))
                    mask1 = mask1.astype('float32') / 255
                    mask_one = np.expand_dims(mask1, 0)

                    masks.append(mask_one)

            img = img.astype('float32') / 1.0
            img = img.transpose(2, 0, 1)
            mask = mask.astype('float32') / 1.0
            mask = mask.transpose(2, 0, 1)



        else:
            mask = []
            masks = []

            for i in range(self.num_classes):
                #i = i+1
                mask_image = cv2.imread(os.path.join(self.mask_dir, str(i),img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
                mask_image = mask_image.astype('float32') / 255.0
                mask_image = mask_image.astype('uint8')
                mask.append(mask_image[..., None])

            mask = np.dstack(mask)
            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            img = img.astype('float32') / 1.0
            img = img.transpose(2, 0, 1)
            mask = mask.astype('float32') / 1.0
            mask = mask.transpose(2, 0, 1)

        return ori_img, img, mask, masks, {'img_id': img_id}


def patch_gen(img, mask, p_size,  overlap = 0.5):

    img_h = img.shape[0]
    img_w = img.shape[1]
    shift_size = 1- overlap
    i_w = int(math.floor((img_w-p_size) / math.ceil(shift_size * p_size))) +1
    i_h = int(math.floor((img_h-p_size) / math.ceil(shift_size * p_size))) +1

    image_patch = []
    mask_patch = []

    h_step = int(math.ceil(shift_size * p_size))
    w_step = int(math.ceil(shift_size * p_size))
    for i in range(i_w):
        for j in range(i_h):
            idx_w1 = int(i*w_step)
            idx_w2 = idx_w1 + p_size
            idx_h1 = int(j*h_step)
            idx_h2 = int(idx_h1 + p_size)
            if idx_h1 < 0 or idx_w1 < 0:
                print('err')
            if idx_h2 > img_h or idx_w2 > img_w:
                print('err')
            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2,:])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2,:])

    for i in range(i_w):
        for j in range(i_h):
            idx_w2 = int(img_w - (i * w_step))
            idx_w1 = int(idx_w2 - p_size)
            idx_h2 = int(img_h - (j*h_step))
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
            idx_h2 = int(img_h - (j*h_step))
            idx_h1 = int(idx_h2 - p_size)
            if (idx_h2- idx_h1) != p_size or (idx_w2- idx_w1) != p_size :
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
            idx_h1 = int(j*h_step)
            idx_h2 = int(idx_h1 + p_size)
            if (idx_h2- idx_h1) != p_size or (idx_w2- idx_w1) != p_size :
                print('err')
            if idx_h1 < 0 or idx_w1 < 0:
                print('err')
            if idx_h2 > img_h or idx_w2 > img_w:
                print('err')

            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2, :])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2, :])


    return image_patch, mask_patch

def post_process_resized_mask(resized_mask):

    mask_1 = ((resized_mask > 125) & (resized_mask < 255))
    resized_mask[mask_1] = 255

    #tmp_maks = np.zeros((resized_mask.shape[0],resized_mask.shape[1]))
    mask_0 = ((resized_mask > 0) & (resized_mask <= 125))
    resized_mask[mask_0] = 0
    #tmp_maks[mask_0] = 1
    #if np.sum(tmp_maks) > 0:
    #    print('0 label error')

    return resized_mask

class DatasetPatch(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_ext, mask_ext, num_classes, input_channels, image_w, psize, patch_overlap, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        #test = os.path.join(img_ids,'*_image.*')
        image_paths = glob(img_ids)
        self.img_paths = image_paths
        #self.img_dir = img_dir
        #self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.infer_size = image_w
        self.psize = psize
        self.patch_overlap = patch_overlap

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        base_name = os.path.basename(img_path)
        label_path = img_path.replace('image', 'labels')

        img_input = cv2.imread( img_path)
        mask_input = cv2.imread(label_path)
        img_patch_set = []
        mask_patch_set = []
        p_size = self.psize
        img_size = self.infer_size
        image_patch, mask_patch = patch_gen(img_input, mask_input, p_size, self.patch_overlap)

        patch_len = len(image_patch)
        for idx in range(patch_len):
            img = image_patch[idx]
            mask = mask_patch[idx]
            img = cv2.resize(img, (img_size, img_size))
            mask = cv2.resize(mask, (img_size, img_size))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.expand_dims(mask, 0)
            mask = mask.transpose(1, 2, 0)
            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']


            img = img.astype('float32') / 1.0
            img = img.transpose(2, 0, 1)
            img_patch_set.append(img)
        img_patch_set = np.array(img_patch_set)
        mask_patch_set = np.array(mask_patch)

        return img_input, mask_input, img_patch_set, mask_patch, {'img_name': base_name}