from mmdet.apis import set_random_seed
from mmcv import Config
#cfg = Config.fromfile('./configs/solov2/solov2_r50_fpn_1x_coco.py')
cfg = Config.fromfile('./configs/solov2/solov2_x101_dcn_fpn_3x_coco.py')
#cfg = Config.fromfile('./configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py')
#cfg = Config.fromfile('./configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')
#cfg = Config.fromfile('./configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py')

#cfg.num_things_classes = 1


cfg.dataset_type = 'CocoDataset'
cfg.img_norm_cfg = dict(
    mean=[0.932 * 255, 0.932 * 255, 0.932 * 255], std=[0.091 * 255, 0.091 * 255, 0.091 * 255])


#cfg.data.train.ann_file = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depthstiffness_coco_train.json'
#cfg.data.train.img_prefix = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/'
#cfg.data.train.classes = ('fore',)

#cfg.data.val.ann_file = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depthstiffness_coco_val.json'
#cfg.data.val.img_prefix = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/'
#cfg.data.val.classes = ('fore',)

#cfg.data.test.ann_file = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depthstiffness_coco_val.json'
#cfg.data.test.img_prefix = '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/'
#cfg.data.test.classes = ('fore',)

cfg.data.train.ann_file = '/home/user/Downloads/dataset_iros2022_v4_mod/dataset/v4_mod_seg_train.json'
cfg.data.train.img_prefix = '/home/user/Downloads/dataset_iros2022_v4_mod/dataset/Depth/'
cfg.data.train.classes = ('fore',)

cfg.data.val.ann_file = '/home/user/Downloads/dataset_iros2022_v4_mod/dataset/v4_mod_seg_val.json'
cfg.data.val.img_prefix = '/home/user/Downloads/dataset_iros2022_v4_mod/dataset/Depth/'
cfg.data.val.classes = ('fore',)

cfg.data.test.ann_file = '/home/user/Downloads/dataset_iros2022_v4_mod/dataset/v4_mod_seg_test.json'
cfg.data.test.img_prefix = '/home/user/Downloads/dataset_iros2022_v4_mod/dataset/Depth/'
cfg.data.test.classes = ('fore',)

cfg.model.mask_head.num_classes = 1

#cfg.load_from = './checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

cfg.work_dir = './tutorial_exps'

#cfg.optimizer.lr = 0.02 / 8
#cfg.lr_config.warmup = None
#cfg.log_config.interval = 10

#cfg.evaluation.metric = 'bbox'
#cfg.evaluation.interval = 2
#cfg.checkpoint_config.interval = 2

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

print(f'Config:\n{cfg.pretty_text}')

cfg.dump('./my_customconfig_veluga.py')