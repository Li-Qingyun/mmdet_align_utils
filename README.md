# glcc22_align_utils

align_state_dict.py  对齐checkpoint的参数字典的工具
  1代表load的源码repo的checkpoint经过 各种map和删除增加等操作 之后的模型状态字典,2代表当前mmdet的模型的状态字典.
  这里主要的工作就是写上面这个 各种map和删除增加等操作, 比如, 对每个参数字典的key写映射的replace, 删除 bn 前的bias (dino源码是有的), 给pooling和bn加num_batch_tracked, 把91类参数映射为80类等操作
  不断修改这些操作的逻辑, 使1的状态字典的names能和2完全一致. names在这里写成json文件, 可以直接pycharm双击shift选择对比差异, 然后两侧加载两个文件, 这样做的好处是, 可以直接点上面的那个小铅笔来直接重新读两个文件, 而不需要每次都复制, 右键与剪切板对比这样麻烦.
