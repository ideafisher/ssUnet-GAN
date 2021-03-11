
import json, os
from glob import glob

from aerial_image_segmentation_api import load_segmentation_models, get_patched_input, segmentation_inference, save_image_color_masking


def main():
    config_file = "../configs/config_v1.json"
    config_dict = json.loads(open(config_file, 'rt').read())
    val_config = config_dict['val_config']
    full_img_path = val_config['full_image_path']
    image_folder = os.path.join(full_img_path, '*_image.*')
    output_folder = '../outputs'

    gt_mask_flag = True
    if 'False' in val_config['gt_mask_flag']:
        gt_mask_flag = False
    else:
        gt_mask_flag = True

    ## Test Image
    image_paths = glob(image_folder)
    img_path = image_paths[0]
    save_image_name = os.path.basename(img_path)
    save_name, ext = os.path.splitext(save_image_name)
    ## Load Segmentation Model
    model, config = load_segmentation_models(config_file)


    for c in range(config['num_classes']):
        os.makedirs(os.path.join( output_folder, config['name'], str(c)), exist_ok=True)

    ## Get patch images
    img_input, img_patch_set, mask_patch_set = get_patched_input(img_path, config, gt_mask_flag)

    ## Operate the segmentation with patches
    re_class_mask, gt_class_mask = segmentation_inference(model, img_input, img_patch_set, mask_patch_set, config, gt_mask_flag)

    ## save the color mask of regions
    save_image_color_masking(output_folder, save_name, img_input, re_class_mask, gt_class_mask, config, gt_mask_flag)


    return



if __name__ == '__main__':
    main()

