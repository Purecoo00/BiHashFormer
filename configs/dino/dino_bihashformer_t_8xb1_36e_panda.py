# BiHashFormer-T with DINO: 36 epochs, eight GPUs, one image per GPU.
_base_ = [
   '../_base_/datasets/panda_detection.py', '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['bihash_former.models'], allow_failed_imports=False)

num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    type='BiHashDINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='BiHashFormer', embed_dims=96, depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24), window_size=21,
        num_bits=32, kv_ratio=0.5, num_prototypes=3, radius=4,
        roi_pool_sizes=(32, 16, 8, 4), select_gamma=0.1,
        with_cp=True, init_cfg=None),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        end_factor=1.0,
        begin=0,
        end=7120,
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10, 20, 30],
        gamma=0.3)
]

auto_scale_lr = dict(base_batch_size=8)

find_unused_parameters=True
