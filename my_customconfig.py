dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[237.66000000000003, 237.66000000000003, 237.66000000000003],
    std=[23.205, 23.205, 23.205])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
                   (1333, 672), (1333, 640)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[237.66000000000003, 237.66000000000003, 237.66000000000003],
        std=[23.205, 23.205, 23.205],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[237.66000000000003, 237.66000000000003, 237.66000000000003],
                std=[23.205, 23.205, 23.205],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depthstiffness_coco_train.json',
        img_prefix=
        '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='Resize',
                img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
                           (1333, 672), (1333, 640)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[237.66000000000003, 237.66000000000003, 237.66000000000003],
                std=[23.205, 23.205, 23.205],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        classes=('fore', )),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depthstiffness_coco_val.json',
        img_prefix=
        '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[237.66000000000003, 237.66000000000003, 237.66000000000003],
                        std=[23.205, 23.205, 23.205],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('fore', )),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depthstiffness_coco_val.json',
        img_prefix=
        '/home/deepstation/Downloads/dataset_iros2022_v4_mod/dataset/Depth/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[237.66000000000003, 237.66000000000003, 237.66000000000003],
                        std=[23.205, 23.205, 23.205],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('fore', )))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='SOLOv2',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'),
        style='pytorch',
        groups=64,
        base_width=4,
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=1,
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            conv_cfg=dict(type='DCNv2')),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=True),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=100))
work_dir = './tutorial_exps'
seed = 0
gpu_ids = range(0, 1)
