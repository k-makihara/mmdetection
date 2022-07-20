import os
import os.path as osp
from PIL import Image
import mmcv
import numpy as np

def convert_stiffness_to_coco(root, inputs, out_file, num_st, num_en):

    data_infos = list(sorted(os.listdir(osp.join(root, inputs))))[num_st:num_en]
    

    data_infos_out = list(sorted(os.listdir(osp.join(root, "Segment"))))[num_st:num_en]

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(data_infos):
        img_path = osp.join(root, inputs, data_infos[idx])
        

        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=v,
            height=height,
            width=width))

        mask_path = osp.join(root, "Segment", data_infos_out[idx])
        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = np.array(masks, dtype=np.int8)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []

        #boxes = np.asarray(boxes, dtype=np.float32)
        #labels = np.ones((num_objs,), dtype=np.int64)
        
        #image_id = np.array([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = np.zeros((num_objs,), dtype=np.int64)


        for i in range(num_objs):
            pos = np.where(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            #boxes.append([xmin, ymin, xmax, ymax])

            px = pos[1]
            py = pos[0]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

  

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'fore'}])
    mmcv.dump(coco_format_json, root+"/"+inputs+out_file)

convert_stiffness_to_coco("/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset", "Depth", "stiffness_coco_train.json", 0, 10000)
convert_stiffness_to_coco("/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset", "Depth", "stiffness_coco_val.json", 0, 100)
