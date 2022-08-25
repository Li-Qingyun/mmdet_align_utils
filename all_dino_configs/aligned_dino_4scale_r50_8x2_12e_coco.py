_base_ = 'dino_4scale_r50_8x2_12e_coco.py'
model = dict(
    backbone=dict(norm_cfg=dict(type='FrozenBatchNorm2d')),
    neck=dict(conv_cfg=dict(type='Conv2d', bias=True)),
    bbox_head=dict(num_classes=91))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(continuous_categories=False),
    val=dict(continuous_categories=False),
    test=dict(continuous_categories=False))