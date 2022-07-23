import os
from configs._base_.datasets.coco_detection import data as data_cfg
from mmdet.datasets.coco import CocoDataset


train_cfg = data_cfg['train']
assert train_cfg.pop('type') == 'CocoDataset'
train_cfg['continuous_categories'] = False

dataset = CocoDataset(**train_cfg)

filename2idx = {data_info['filename']: idx
                for idx, data_info in enumerate(dataset.data_infos)}


def get_ann(filename: str):
    idx = filename2idx[filename]
    return dataset.get_ann_info(idx)