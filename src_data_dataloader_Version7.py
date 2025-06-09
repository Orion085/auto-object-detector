import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ObjectDetectionDataset(Dataset):
    """
    Generic dataset for object detection.
    Expects a directory of images and a JSON annotation file (COCO-style).
    """
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        anns = self.annotations.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append(bbox)
            labels.append(ann['category_id'])

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target