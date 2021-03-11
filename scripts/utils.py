import argparse
import cv2
import numpy as np

def adjust_brightness(img):
    ## brightnesss full image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    box_h, box_s, box_v = cv2.split(hsv)
    avg_h_color = np.average(box_h)
    avg_s_color = np.average(box_s)
    avg_v_color = np.average(box_v)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    ###
    '''
    stand_h = 110
    diff_h = stand_h - avg_h_color
    h = h + diff_h
    h = np.clip(np.round(h), 1, 255)
    new_h = h.astype(np.uint8)
    '''
    ###
    '''
    stand_s = 140
    diff_s = stand_s - avg_s_color
    s = s + diff_s
    s = np.clip(np.round(s), 1, 255)
    new_s = s.astype(np.uint8)
    '''
    ###
    stand_b = 90
    # diff_b = stand_b - avg_v
    diff_b = stand_b - avg_v_color
    v = v + diff_b
    v = np.clip(np.round(v), 1, 255)
    new_v = v.astype(np.uint8)

    final_hsv = cv2.merge((h, s, new_v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
