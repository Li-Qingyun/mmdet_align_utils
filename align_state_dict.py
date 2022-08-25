# original repo: https://github.com/IDEACVR/DINO
import json
from collections import OrderedDict

import torch

from mmcv import Config
from mmcv.runner import save_checkpoint
from mmdet.utils import replace_cfg_vals
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, CocoDataset
from mmdet.models.utils.ckpt_convert import swin_converter


## Frequent settings
CKPT_IDX = 9
PIPELINE = dict(
    delete_duplicated_prediction_head=True,  # for two-stage DeformDETR-like model
    delete_bias_before_norm_in_neck=True,
    # mapping_state_dict
    add_num_batches_tracked=True,  # default to set 0
    map_91_cls_to_80_cls=True
)
IS_ALIGNED = False


## Infrequent settings
USER_DIR = '/home/' + 'rs'
CKPT_DIR = USER_DIR + r'/Desktop/DINO/ckpts/'
ckpt_filenames = [
    'checkpoint0011_4scale.pth',
    'checkpoint0023_4scale.pth',
    'checkpoint0033_4scale.pth',
    'checkpoint0011_5scale.pth',
    'checkpoint0022_5scale.pth',
    'checkpoint0031_5scale.pth',
    'checkpoint0011_4scale_swin.pth',
    'checkpoint0029_4scale_swin.pth',
    'checkpoint0011_5scale_swin.pth',
    'checkpoint0027_5scale_swin.pth',
    # 'checkpoint_4scale_42seedinit.pth',
    # 'checkpoint_5scale_42seedinit.pth'
    ]
IS_INITED_CKPT = CKPT_IDX > 9
file_path_list = [CKPT_DIR + filename for filename in ckpt_filenames]

NUM_SCALE = int(ckpt_filenames[CKPT_IDX][
                ckpt_filenames[CKPT_IDX].index('scale') - 1])
BATCH = '8x2' if NUM_SCALE <= 4 else '16x1'
BACKBONE = 'swin-l-p4-w12' if 'swin' in ckpt_filenames[CKPT_IDX] else 'r50'
CFG_PATH = USER_DIR + r'/Desktop/DINO_mmdet/configs' \
           rf'/dino/dino_{NUM_SCALE}scale_{BACKBONE}_{BATCH}_12e_coco.py'

if IS_ALIGNED:
    PIPELINE['delete_bias_before_norm_in_neck'] = False
    PIPELINE['map_91_cls_to_80_cls'] = False
    PIPELINE['map_label_embeddings_to_91_cls'] = True
    tmp = CFG_PATH.split('/')
    CFG_PATH = '/'.join(tmp[:-1]) + '/aligned_' + tmp[-1]


def get_mapped_name(name: str):
    new_name = name
    if new_name.startswith('backbone'):
        new_name = new_name.replace('backbone.0.body', 'backbone')
    if 'transformer' in new_name:
        new_name = 'bbox_head.' + new_name

    if 'input_proj' in new_name:
        components = new_name.split('.')
        lid, sublid = components[-3:-1]
        components[-2] = 'gn' if sublid == '1' else 'conv'
        if int(lid) < NUM_SCALE - 1:
            components[0] = 'neck.convs'
        else:
            components[0] = 'neck.extra_convs'
            components[1] = '0'
        new_name = '.'.join(components)

    if new_name.startswith('class_embed'):
        new_name = new_name.replace(
            'class_embed', 'bbox_head.cls_branches')
    if new_name.startswith('bbox_embed'):
        components = new_name.split('.')
        lid, _, sublid = components[1:4]
        components[3] = {'0': '0', '1': '2', '2': '4'}[sublid]
        del components[2]
        components[0] = 'bbox_head.reg_branches'
        new_name = '.'.join(components)
    if 'enc_out_' in new_name:
        new_name = new_name.replace('transformer.enc_out_class_embed',
                                    'cls_branches.6')
        new_name = new_name.replace('transformer.enc_out_bbox_embed.layers.0',
                                    'reg_branches.6.0')
        new_name = new_name.replace('transformer.enc_out_bbox_embed.layers.1',
                                    'reg_branches.6.2')
        new_name = new_name.replace('transformer.enc_out_bbox_embed.layers.2',
                                    'reg_branches.6.4')

    if 'transformer.encoder.layers' in new_name:
        new_name = new_name.replace('self_attn', 'attentions.0')
        new_name = new_name.replace('norm1', 'norms.0')
        new_name = new_name.replace('norm2', 'norms.1')
        new_name = new_name.replace('linear1', 'ffns.0.layers.0.0')
        new_name = new_name.replace('linear2', 'ffns.0.layers.1')
    if 'transformer.decoder.layers' in new_name:
        new_name = new_name.replace('self_attn', 'attentions.0')
        new_name = new_name.replace('cross_attn', 'attentions.1')
        new_name = new_name.replace('norm1', 'norms.0')
        new_name = new_name.replace('norm2', 'norms.1')
        new_name = new_name.replace('linear1', 'ffns.0.layers.0.0')
        new_name = new_name.replace('linear2', 'ffns.0.layers.1')
        new_name = new_name.replace('in_proj_weight', 'attn.in_proj_weight')
        new_name = new_name.replace('out_proj.weight', 'attn.out_proj.weight')
        new_name = new_name.replace('in_proj_bias', 'attn.in_proj_bias')
        new_name = new_name.replace('out_proj.bias', 'attn.out_proj.bias')
        new_name = new_name.replace('norm3', 'norms.2')

    if 'ref_point_head' in new_name:
        new_name = new_name.replace('ref_point_head.layers.0',
                                    'ref_point_head.0')
        new_name = new_name.replace('ref_point_head.layers.1',
                                    'ref_point_head.2')

    if new_name.startswith('label_enc'):
        new_name = 'bbox_head.label_embedding.weight'
    if new_name.endswith('level_embed'):
        new_name = new_name + 's'
    if 'tgt_embed' in new_name:
        new_name = new_name.replace('tgt_embed', 'query_embed')

    # false order of norm in original repo
    if new_name.startswith('bbox_head.transformer.decoder.layers') \
            and 'norms' in new_name:
        false_name = new_name
        if 'norms.0.' in new_name:
            new_name = new_name.replace('norms.0', 'norms.1')
            print(f'rename false order of {false_name} to {new_name}')
        elif 'norms.1' in new_name:
            new_name = new_name.replace('norms.1', 'norms.0')
            print(f'rename false order of {false_name} to {new_name}')
        else:
            new_name = new_name

    return new_name


def map_cls(state_dict: OrderedDict, dataset: CocoDataset, num_classes):
    if num_classes == 80:
        for lid in range(7):
            l_name = f'bbox_head.cls_branches.{lid}.weight'
            state_dict[l_name] = state_dict[l_name][dataset.cat_ids, :]
            l_name = f'bbox_head.cls_branches.{lid}.bias'
            state_dict[l_name] = state_dict[l_name][dataset.cat_ids]
        l_name = 'bbox_head.label_embedding.weight'
        label_embedding_indices = dataset.cat_ids
        state_dict[l_name] = state_dict[l_name][label_embedding_indices, :]
    elif num_classes == 91:
        l_name = 'bbox_head.label_embedding.weight'
        state_dict[l_name] = state_dict[l_name][:-1, :]
    else:
        raise NotImplementedError
    return state_dict


def mapping_state_dict(state_dict: OrderedDict):
    out = OrderedDict()
    for name, param in state_dict.items():
        new_name = get_mapped_name(name)
        assert new_name not in out, f'{name}-->{new_name}'
        out[new_name] = param
    return out


def add_num_batches_tracked(state_dict: OrderedDict):
    _keys = list()
    for name, param in state_dict.items():
        if 'bn' in name and 'weight' in name:
            _keys.append(
                name[:name.index('bn') + 3] + '.num_batches_tracked')
        if 'downsample.1' in name and 'weight' in name:
            _keys.append(
                name[:name.index('downsample.1') + 12] + '.num_batches_tracked')
    for name in _keys:
        state_dict[name] = torch.tensor([0], dtype=torch.int64)
    return state_dict


def delete_duplicated_items(state_dict: OrderedDict):
    out = OrderedDict()
    duplicated_keys = []
    for name, param in state_dict.items():
        if name.startswith('bbox_embed') or name.startswith('class_embed'):
            duplicated_key = 'transformer.decoder.' + name
            assert torch.equal(param, state_dict[duplicated_key])
            print(f'{duplicated_key} is deleted as duplicated items')
            duplicated_keys.append(duplicated_key)
    for name, param in state_dict.items():
        if name not in duplicated_keys:
            out[name] = param
    assert len(state_dict) - len(out) == len(duplicated_keys)
    return out


def delete_bias_before_norm(state_dict: OrderedDict):
    for i in range(4):
        del state_dict[f'input_proj.{i}.0.bias']
        print(f'input_proj.{i}.0.bias is deleted as bias before norm')
    return state_dict


if __name__ == '__main__':
    # load and convert ckpt released in original repo
    ckpt_1 = torch.load(file_path_list[CKPT_IDX])
    model_1 = ckpt_1.get('model', ckpt_1.get('inited_model'))
    if 'swin' in BACKBONE:
        _len = len(model_1)
        swin = {k[11:]: model_1.pop(k)
                for k in list(model_1.keys()) if 'backbone.0.' in k}
        swin = swin_converter(swin)
        model_1.update(swin)
        assert _len == len(model_1)
    if PIPELINE['delete_duplicated_prediction_head']:
        model_1 = delete_duplicated_items(model_1)
    if PIPELINE['delete_bias_before_norm_in_neck']:
        model_1 = delete_bias_before_norm(model_1)
    model_1 = mapping_state_dict(model_1)
    if PIPELINE['add_num_batches_tracked']:
        model_1 = add_num_batches_tracked(model_1)

    # build model of our implementation with mmdet
    cfg = replace_cfg_vals(Config.fromfile(CFG_PATH))
    cfg.model.train_cfg = None
    dino_mmdet = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    dataset = build_dataset(cfg.data.test)
    class_mapping = dataset.cat_ids
    torch.save(class_mapping, r'./developing/class_mapping')

    model_2 = dino_mmdet.state_dict()
    if PIPELINE['map_91_cls_to_80_cls'] \
            or PIPELINE.get('map_label_embeddings_to_91_cls', False):
        model_1 = map_cls(model_1, dataset, cfg.model.bbox_head.num_classes)

    names_1 = sorted([name for name, _ in model_1.items()])
    names_2 = sorted([name for name, _ in model_2.items()])
    json.dump(names_2, open(r'./developing/names_2.json', 'w'), indent=0)
    json.dump(names_1, open(r'./developing/names_1.json', 'w'), indent=0)

    # At last:
    print(dino_mmdet.load_state_dict(model_1, strict=False))
    save_path = f'./developing/' \
                f'{"aligned_" if IS_ALIGNED else ""}' \
                f'{ckpt_filenames[CKPT_IDX][:-4]}_mmdet.pth'
    # save_checkpoint(dino_mmdet, save_path,
    #                 meta=dict(epoch=0, iter=0, convert_pipeline=PIPELINE))
    torch.save(dict(state_dict=model_1, meta=dict(
        epoch=0, iter=0, convert_pipeline=PIPELINE)), save_path)
    print(f'The converted ckpt has been saved at {save_path}')