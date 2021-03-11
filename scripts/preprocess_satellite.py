import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

def save_contour(img, mask):

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(mask, 127, 255, 0)
    val = np.max(img_binary)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (255, 0, 0), 3)

    result_dir = './checkpoint/result'
    name = os.path.join(result_dir, 'result_dir_{:d}.png'.format(2))
    cv2.imwrite(name, img)

def patch_gen(img, mask, p_size):

    img_h = img.shape[0]
    img_w = img.shape[1]
    overlap = 0.5
    i_w = int(math.floor(img_w / (overlap * p_size)))-1
    i_h = int(math.floor(img_h / (overlap * p_size)))-1

    image_patch = []
    mask_patch = []

    h_step = int(overlap * p_size)
    w_step = int(overlap * p_size)
    for i in range(i_w):
        for j in range(i_h):
            idx_w1 = int(i*w_step)
            idx_w2 = idx_w1 + p_size
            idx_h1 = int(j*h_step)
            idx_h2 = int(idx_h1 + p_size)
            #print(idx_w1, idx_w2, idx_h1, idx_h2)
            image_patch.append(img[idx_h1:idx_h2, idx_w1:idx_w2,:])
            mask_patch.append(mask[idx_h1:idx_h2, idx_w1:idx_w2,:])

    for i in range(i_w):
        for j in range(i_h):
            idx_w2 = int(img_w - (i * w_step))
            idx_w1 = int(idx_w2 - p_size)

            idx_h2 = int(img_h - (j*h_step))
            idx_h1 = int(idx_h2 - p_size)
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
def save_image_mask(image_paths, dataset_node, image_name, num_class, img_size):
    data_cnt = 0
    for i in tqdm(range(len(image_paths))):
        img_path = image_paths[i]
        label_path = img_path.replace('image','labels')
        img = cv2.imread(img_path)
        mask_img_or = cv2.imread(label_path)

        #mask_img = cv2.cvtColor(mask_img_or, cv2.COLOR_BGR2GRAY)
        p_size = 1024
        image_patch, mask_patch = patch_gen(img, mask_img_or, p_size)
        for pidx in range(len(image_patch)):
            p_image = image_patch[pidx]
            p_mask = mask_patch[pidx]
            data_cnt += 1
        #resized_img = cv2.resize(img, (img_size, img_size))
            file_name = '{:s}_{:05d}'.format(image_name, data_cnt) + '.png'

            path = os.path.join('../inputs/{:s}_{:d}'.format(image_name, img_size),'images', dataset_node, file_name)
            resized_p_image = cv2.resize(p_image, (img_size, img_size))
            cv2.imwrite(path, resized_p_image)

        #mask_img = cv2.cvtColor(mask_img_or, cv2.COLOR_BGR2GRAY)

            all_mask = np.zeros((img_size,img_size))
            for idx in range(num_class):
                mask = np.zeros((p_image.shape[0], p_image.shape[1]))
                if idx == 0:
                    mask_ = ((p_mask[:, :, 0] == 255) & (p_mask[:, :, 1] == 255) & (p_mask[:, :, 2] == 255))
                if idx == 1:
                    mask_ = (( p_mask[:,:,0] == 255) & ( p_mask[:,:,1] == 0) & ( p_mask[:,:,2] == 0))

                if idx == 2:
                    mask_ = ((p_mask[:, :, 0] == 0) & (p_mask[:, :, 1] == 0) & (p_mask[:, :, 2] == 255))

                mask[mask_] = 1
                #print(np.sum(mask))

                mask = (mask * 255.0).astype('uint8')
                resized_mask = cv2.resize(mask, (img_size, img_size))
                if 1:
                    resized_mask = post_process_resized_mask(resized_mask)
                #resized_mask = (resized_mask / 255.0).astype('uint8')
                path = os.path.join('../inputs/{:s}_{:d}/'.format(image_name, img_size),'annotations', dataset_node, str(idx), file_name)
                single_label = resized_mask >0
                all_mask[single_label] = idx
                cv2.imwrite(path, resized_mask)

            path = os.path.join('../inputs/{:s}_{:d}'.format(image_name, img_size), 'annotations', dataset_node,  file_name)
            all_mask = (all_mask * 1).astype('uint8')
            cv2.imwrite(path, all_mask)
def main():
    img_size = 512

    image_name = 'chicago'
    image_paths = glob('../inputs/{:s}/*_image.*'.format(image_name))
    base_paths = os.path.join('inputs',image_name)

    os.makedirs('../inputs/{:s}_{:d}/images'.format(image_name,img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/masks'.format(image_name, img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/images/training'.format(image_name,img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/annotations/training'.format(image_name, img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/images/validation'.format(image_name,img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/annotations/validation'.format(image_name, img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/images/test'.format(image_name,img_size), exist_ok=True)
    os.makedirs('../inputs/{:s}_{:d}/annotations/test'.format(image_name, img_size), exist_ok=True)

    train_image_path, val_test_image_path = train_test_split(image_paths, test_size=0.2, random_state=41)
    val_image_path, test_image_path = train_test_split(val_test_image_path, test_size=0.5, random_state=41)

    single_mask = False
    if single_mask != True:
        num_class = 3
        for idx in range(num_class):
            base_path = '../inputs/{:s}_{:d}/annotations'.format(image_name, img_size)
            path = os.path.join(base_path,'training',str(idx))
            os.makedirs(path, exist_ok=True)
            path = os.path.join(base_path,'validation', str(idx))
            os.makedirs(path, exist_ok=True)
            path = os.path.join(base_path, 'test', str(idx))
            os.makedirs(path, exist_ok=True)

    dataset_node ='training'
    save_image_mask(train_image_path, dataset_node, image_name, num_class, img_size)

    dataset_node = 'validation'
    save_image_mask(val_image_path, dataset_node, image_name, num_class, img_size)

    dataset_node = 'test'
    save_image_mask(test_image_path, dataset_node, image_name, num_class, img_size)

def make_data_list():
    tr_image = glob('../inputs/aerial/images/training/*.*')
    v_image = glob('../inputs/aerial/images/validation/*.*')
    t_image = glob('../inputs/aerial/images/test/*.*')

    tr_annotations = glob('../inputs/aerial/annotations/training/*.*')
    v_annotations = glob('../inputs/aerial/annotations/validation/*.*')
    t_annotations = glob('../inputs/aerial/annotations/test/*.*')

    list_path = '../inputs/aerial/list'
    os.makedirs(list_path, exist_ok=True)

    f = open("../inputs/aerial/list/test.txt", 'w')
    v_img_path = 'images/test/'
    v_ann_path = 'annotations/test/'
    for i_path, a_path in zip(t_image, t_annotations):
        i_name = os.path.basename(str(i_path))
        a_name = os.path.basename(str(a_path))
        img_path = os.path.join(v_img_path, i_name)
        ann_path = os.path.join(v_ann_path, a_name)
        f_set = img_path+' '+ann_path
        f.write("%s\n" % f_set)
    f.close()


    f = open("../inputs/aerial/list/validation.txt", 'w')
    v_img_path = 'images/validation/'
    v_ann_path = 'annotations/validation/'
    for i_path, a_path in zip(v_image, v_annotations):
        i_name = os.path.basename(str(i_path))
        a_name = os.path.basename(str(a_path))
        img_path = os.path.join(v_img_path, i_name)
        ann_path = os.path.join(v_ann_path, a_name)
        f_set = img_path+' '+ann_path
        f.write("%s\n" % f_set)
    f.close()

    f = open("../inputs/aerial/list/training.txt", 'w')
    v_img_path = 'images/training/'
    v_ann_path = 'annotations/training/'
    for i_path, a_path in zip(tr_image, tr_annotations):
        i_name = os.path.basename(str(i_path))
        a_name = os.path.basename(str(a_path))
        img_path = os.path.join(v_img_path, i_name)
        ann_path = os.path.join(v_ann_path, a_name)
        f_set = img_path+' '+ann_path
        f.write("%s\n" % f_set)
    f.close()

    return 0

if __name__ == '__main__':
    #main()
    make_data_list()
