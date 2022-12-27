from dataset.transform import *
from dataset.xml_to_dict import xml2dict

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
        assert layout_txt is not None, 'Unknown mode'
    
        self.transform = transform
        self.get_info = get_info

        self.image_list = []
        self.annotation_list = []
        for r, txt in zip(root, layout_txt):
            self.image_list += [os.path.join(r, 'JPEGImages', t = '.jpg') for t in get_file_names(r, txt)]
            self.annotation_list += [os.path.join(r, 'Annotations', t + '.xml') for t in get_file_names(r, txt)]

        def __len__(self):
            return len(self.annotation_list)

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

if __name__ == "__main__":
    