import json
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

def xml2dict(xml):
    data = {c.tag: None for c in xml}
    for c in xml:
        def add(data, tag, text):
            if data[tag] is None:
                data[tag] = text
            elif isinstance(data[tag], list):
                data[tag].append(text)
            else:
                data[tag] = [data[tag], text]
            return data

        if len(c) == 0:
            data = add(data, c.tag, c.text)
        else:
            data = add(data, c.tag, xml2dict(c))
    return data

root = r'/home/tianqijian/datasets/VOC2007/VOCtrainval_06-Nov-2007/VOC2007'
# construct the path for each annotation
annotation_root = os.path.join(root, 'Annotations')
annotation_list = os.listdir(annotation_root)
annotation_list = [os.path.join(annotation_root, a) for a in annotation_list]

s = set()
for a in tqdm(annotation_list):
    xml = ET.parse(os.path.join(a)).getroot()
    data = xml2dict(xml)['object']
    # one picture has multi objects
    if isinstance(data, list):
        for d in data:
            s.add(d['name'])
    else:
        s.add(data['name'])

s = list(s)
s.sort()

data = {value: i for i, value in enumerate(s)}
json_str = json.dumps(data)
json_path = './classes.json'

with open(json_path, 'w') as f:
    f.write(json_str)

