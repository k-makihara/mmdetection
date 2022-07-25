# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils


def collect_files(img_dir, gt_dir, num_s, num_e):
    suffix = '.png'
    files = []
    for img_file in glob.glob(osp.join(img_dir, '*.png'))[num_s:num_e]:
        assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + img_file[
            len(img_dir):]
        files.append((img_file, inst_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, inst_file = files
    inst_img = mmcv.imread(inst_file, 'unchanged')
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    unique_inst_ids = np.unique(inst_img[inst_img > 0])
    anno_info = []
    for inst_id in unique_inst_ids:
        # For non-crowd annotations, inst_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        iscrowd = 0
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        mask_rle['counts'] = mask_rle['counts'].decode()


        anno = dict(
            iscrowd=iscrowd,
            category_id="fore",
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(img_file))
    video_name_seg = osp.basename(osp.dirname(inst_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(img_file)),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(video_name_seg, osp.basename(inst_file)))

    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1

    out_json['categories'].append(dict(id=0, name="fore"))

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    #parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = "/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset"
    out_dir = cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, "Depth")
    gt_dir = osp.join(cityscapes_path, "Segment")

    set_name = dict(
        train='v4_mod_seg_train.json',
        val='v4_mod_seg_val.json',
        test='v4_mod_seg_test.json')

    num_s = 0
    num_e = 10
    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It took {}s to convert Cityscapes annotation'):
            if split == "train":
                num_s = 0
                num_e = 8000
            elif split == "val":
                num_s = 8000
                num_e = 10000
            elif split == "test":
                num_s = 8000
                num_e = 10000
            files = collect_files(
                osp.join(img_dir), osp.join(gt_dir), num_s, num_e)
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
