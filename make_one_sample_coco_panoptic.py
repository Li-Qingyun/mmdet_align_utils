import json
from tqdm import tqdm


def prepare_coco_onesample_annotation(original_path, sample_id=None):
    coco_json = json.load(open(original_path, 'r'))
    if sample_id is None:
        sample_id = coco_json['annotations'][0]['image_id']
    new_path = f'{original_path[:-5]}_onesample_{sample_id}.json'

    del_indices = []
    for i, item in tqdm(enumerate(coco_json['images'])):
        if item['id'] != sample_id:
            del_indices.append(i)
    del_indices.reverse()
    for indice in tqdm(del_indices):
        coco_json['images'].pop(indice)

    del_indices = []
    for i, item in tqdm(enumerate(coco_json['annotations'])):
        if item['image_id'] != sample_id:
            del_indices.append(i)
    del_indices.reverse()
    for indice in tqdm(del_indices):
        coco_json['annotations'].pop(indice)

    json.dump(coco_json, open(new_path, 'w'))
    print(f'{new_path} has been saved.')
    return sample_id


def prepare_coco_panoptic(train_or_val, sample_id=None):
    sample_id = prepare_coco_onesample_annotation(f'panoptic_{train_or_val}2017.json', sample_id)
    prepare_coco_onesample_annotation(f'instances_{train_or_val}2017.json', sample_id)


def prepare_all_coco_panoptic():
    prepare_coco_panoptic('train')
    prepare_coco_panoptic('val')


if __name__ == '__main__':
    prepare_all_coco_panoptic()
    
