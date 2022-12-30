from dataset.transform import *
from dataset.xml_to_dict import xml2dict
from dataset.draw_bbox import draw

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image

def get_file_names(root, layout_txt):
    with open(os.path.join(root, layout_txt)) as layout_txt:
        file_names = layout_txt.read().split('\n')[:-1]
    return file_names

class VOC2007and2012Dataset(Dataset):
    def __init__(self, root, classes_path, transform, mode, data_range=None, get_info=False):
        with open(classes_path, 'r') as f:
            json_str = f.read()
            self.classes = json.loads(json_str)
        layout_txt = None
        if mode == 'train':
            root = [root[0], root[0], root[1], root[1]]
            layout_txt = [r'ImageSets/Main/train.txt', 
                          r'ImageSets/Main/val.txt',
                          r'ImageSets/Main/train.txt',
                          r'ImageSets/Main/val.txt']
        elif mode == 'test':
            if not isinstance(root, list):
                root = [root]
            layout_txt = [r'ImageSets/Main/test.txt']
        elif mode == 'val':
            root = [root[0], root[1]]
            layout_txt = [r'ImageSets/Main/val.txt',
                          r'ImageSets/Main/val.txt']
        assert layout_txt is not None, 'Unknown mode'
    
        self.transform = transform
        self.get_info = get_info

        self.image_list = []
        self.annotation_list = []
        for r, txt in zip(root, layout_txt):
            self.image_list += [os.path.join(r, 'JPEGImages', t + '.jpg') for t in get_file_names(r, txt)]
            self.annotation_list += [os.path.join(r, 'Annotations', t + '.xml') for t in get_file_names(r, txt)]

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        image_size = image.size
        label = self.label_process(self.annotation_list[idx])

        if self.transform is not None:
            image, label = self.transform(image, label)
            
        if self.get_info:
            return image, label, os.path.basename(self.image_list[idx]).split('.')[0], image_size
        else:
            return image, label  

    def label_process(self, annotation):
        xml = ET.parse(os.path.join(annotation)).getroot()
        data = xml2dict(xml)['object']
        if isinstance(data, list):
            label = [[float(d['bndbox']['xmin']),
                      float(d['bndbox']['ymin']),
                      float(d['bndbox']['xmax']),
                      float(d['bndbox']['ymax']),
                      self.classes[d['name']]]
                    for d in data]
        else:
            label = [[float(data['bndbox']['xmin']),
                      float(data['bndbox']['ymin']),
                      float(data['bndbox']['xmax']),
                      float(data['bndbox']['ymax']),
                      self.classes[data['name']]]]
        label = np.array(label)
        return label
  

if __name__ == "__main__":
    roots = [r'/home/tianqijian/datasets/VOC2007/VOCtrainval_06-Nov-2007/VOC2007',
             r'/home/tianqijian/datasets/VOC2012/VOCtrainval_11-May-2012/VOC2012']
    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
        Resize(448)
    ])
    dataset = VOC2007and2012Dataset(roots, 'classes.json', transforms, 'train', get_info=True)
    print(len(dataset))
    # check ten images randomly
    for i, (image, label, image_name, image_size) in enumerate(dataset):
        if i <= 1000:
            continue
        elif i >= 1010:
            break
        else:
            print(label.dtype)
            print(tuple(image.size()[1:]))
            draw(image, label, dataset.classes)
    print("VOC2007dataset")