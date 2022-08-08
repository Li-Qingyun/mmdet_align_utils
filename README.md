# glcc22_align_utils

该 repo 是我在 glcc22 中进行项目开发时对齐模型精度所使用的脚本工具，目前在向 mmdet PR DINO算法，PR地址为：https://github.com/open-mmlab/mmdetection/pull/8362

## 对齐checkpoint的参数字典的工具 align_state_dict.py  
- 1代表load的源码repo的checkpoint经过 各种map和删除增加等操作 之后的模型状态字典,2代表当前mmdet的模型的状态字典.
- 这里主要的工作就是写上面这个 各种map和删除增加等操作, 比如, 对每个参数字典的key写映射的replace, 删除 bn 前的bias (dino源码是有的), 给pooling和bn加num_batch_tracked, 把91类参数映射为80类等操作
- 不断修改这些操作的逻辑, 使1的状态字典的names能和2完全一致. names在这里写成json文件, 可以直接pycharm双击shift选择对比差异, 然后两侧加载两个文件, 这样做的好处是, 可以直接点上面的那个小铅笔来直接重新读两个文件, 而不需要每次都复制, 右键与剪切板对比这样麻烦.
- 最后只要names完全一致, load能通过(打印出来unexpect_keys和missing_keys检查), 就可以直接获得匹配名称后的 mmdet checkpoint 啦
- 这样的好处是, 完成对齐前这个脚本是用来对齐的工具, 完成对齐后是用来转换源码的checkpoint的工具. 在对齐训练的时候也可以拿来对齐初始化

## mmdetection的coco想要像detr源码一样用91类作为num_classes时 coco.py datasetdevelop.py
- 改mmdet里的coco.py, 让数据集能输出91类的标签
- 如果算法使用的是focal loss, 在train的命令后加: `--options data.train.continuous_categories=False model.bbox_head.num_classes=91` 就可以啦
- datasetdevelop.py 是用来检验这个参数好不好使
- 220726 新增 filter_not_empty_gt 参数, 选择为True后, 会只筛选出不包含目标的样本. 用于在开发阶段debug没有目标的样本的情况. (在DINO中出现了没有目标时loss_dict缺少dn的几项, 导致触发分布式训练的assert)

## 制作小的coco子集 make_one_sample_coco.py 
- 能做一个 一个样本的 coco ann, 需要给样本id
- 能做一个 前n个样本的 coco ann, 需要给样本数

## 对齐参数初始化 align_param_init.py
- 开始我以为所有的参数初始化都能对齐来着, 其实应该直接加载ckpt就行了, 但对于一些源码中特地初始化的参数, 可以把后面的条件加上, 检查这些参数对齐也能避免最后结果不同是参数初始化的原因

## 跳过数据预处理,将源码的数据保存下来用于在mmdet中加载  engine.py 是源码端修改示例
下面是 base.py是mmdet端修改示例
```Python
    def train_step(self, data, optimizer):
        if False:
            img_id = int(data['img_metas'][0]['ori_filename'][:-4])
            data_origin_path = f'developing/data_origin/data_origin_id={img_id}.pth'
            data_origin = torch.load(data_origin_path)
            data_origin['img'] = data_origin['img'].cuda()
            data_origin['gt_bboxes'][0] = data_origin['gt_bboxes'][0].cuda()
            data_origin['gt_labels'][0] = data_origin['gt_labels'][0].cuda()
            data.update(data_origin)

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
```
