import json

def make_one_sample_ann_file(original_ann_path: str, sample_id: int):
    index_images = []
    coco_dataset = json.load(open(original_ann_path))
    original_dataset = coco_dataset.copy()
    for i, img in enumerate(coco_dataset['images']):
        if img['id'] == sample_id:
            index_images.append(i)

    index_annotations = []
    for i, ann in enumerate(coco_dataset['annotations']):
        if ann['image_id'] == sample_id:
            index_annotations.append(i)

    coco_dataset['images'] = [
        coco_dataset['images'][index] for index in index_images]
    coco_dataset['annotations'] = [
        coco_dataset['annotations'][index] for index in index_annotations]

    out_path = f'{original_ann_path[:-5]}_onesample_{sample_id}.json'
    print(f'save one sample ann file at {out_path}')
    json.dump(coco_dataset, open(out_path, 'w'))
    return coco_dataset, original_dataset


def make_n_sample_ann_file(original_ann_path: str, n: int):
    index_images = []
    image_ids = []
    coco_dataset = json.load(open(original_ann_path))
    original_dataset = coco_dataset.copy()
    for i, img in enumerate(coco_dataset['images']):
        if i < n:
            index_images.append(i)
            image_ids.append(img['id'])

    index_annotations = []
    for i, ann in enumerate(coco_dataset['annotations']):
        if ann['image_id'] in image_ids:
            index_annotations.append(i)

    coco_dataset['images'] = [
        coco_dataset['images'][index] for index in index_images]
    coco_dataset['annotations'] = [
        coco_dataset['annotations'][index] for index in index_annotations]

    out_path = f'{original_ann_path[:-5]}_{n}samples.json'
    print(f'save one sample ann file at {out_path}')
    json.dump(coco_dataset, open(out_path, 'w'))
    return coco_dataset, original_dataset


if __name__ == '__main__':
    VAL_SAMPLE_ID = 397133
    VAL_PATH = r'../data/coco/annotations/instances_val2017.json'
    TRAIN_SAMPLE_ID = 391895
    TRAIN_PATH = r'../data/coco/annotations/instances_train2017.json'

    _, val_set = make_one_sample_ann_file(VAL_PATH, VAL_SAMPLE_ID)
    _, train_set = make_one_sample_ann_file(TRAIN_PATH, TRAIN_SAMPLE_ID)
    n_sample_ann, _ = make_n_sample_ann_file(TRAIN_PATH, 2)