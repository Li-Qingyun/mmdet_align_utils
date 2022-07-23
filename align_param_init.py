# repo: https://github.com/IDEACVR/DINO
import json
import torch
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.utils import replace_cfg_vals
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

from developing.align_state_dict import (delete_duplicated_items,
                                         mapping_state_dict, map_cls,
                                         add_num_batches_tracked)


CKPT_IDX = 0
INITED_CKPT_DIR = r'/home/lqy/Desktop/DINO_mmdet/developing/inited_ckpt/'
CFG_PATH_4SCALE = r'/home/lqy/Desktop/DINO_mmdet/configs' \
                  r'/dino/dino_4scale_r50_16x2_12e_coco.py'
CFG_PATH_5SCALE = r'/home/lqy/Desktop/DINO_mmdet/configs' \
                  r'/dino/dino_5scale_r50_16x2_12e_coco.py'

inited_ckpt_filenames = [
    'checkpoint_4scale_42seedinit.pth',
    'checkpoint_5scale_42seedinit.pth'
]

cfg_path = [CFG_PATH_4SCALE, CFG_PATH_5SCALE]

file_path_list = [INITED_CKPT_DIR + filename
                  for filename in inited_ckpt_filenames]


def reset_cfg_num_classes(cfg):
    cfg.model.bbox_head.num_classes = 91
    return cfg


# 1: model of Original DINO repo
ckpt_1 = torch.load(file_path_list[CKPT_IDX])
model_1 = ckpt_1['inited_model']
model_1 = delete_duplicated_items(model_1)
model_1 = mapping_state_dict(model_1)
# model_1 = add_num_batches_tracked(model_1)

# 2: model of mmdet reimplemented DINO
cfg = replace_cfg_vals(Config.fromfile(cfg_path[CKPT_IDX]))
cfg = reset_cfg_num_classes(cfg)
set_random_seed(42, deterministic=True)
dino_mmdet = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
dino_mmdet.init_weights()
dataset = build_dataset(cfg.data.train)

model_2 = dino_mmdet.state_dict()

names_1 = sorted([name for name, _ in model_1.items()])
names_2 = sorted([name for name, _ in model_2.items()])
assert names_1 == names_2

# compare
print('\n\n' + '#'*100)
for name in names_1:

    if not (
        ('backbone' in name)
        or ('bbox_head.reg_branches.6.' in name)
        or ('bbox_head.cls_branches.' in name and 'bias' in name)
    ):
        continue

    param_1 = model_1[name]
    param_2 = model_2[name]
    assert torch.equal(param_1, param_2), name
    print(f'{name} pass through the equal assert')