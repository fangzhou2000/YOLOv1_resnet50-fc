import numpy as np
import torch
import torch.nn as nn
import torchvision

from model.darknet import DarkNet
from model.resnet import yolo_resnet


class yolo(nn.Module):
    """
    s: 7*7
    """
    def __init__(self, s, cell_out_ch, backbone_name, pretrain=None):
        super(yolo, self).__init__()
        self.s = s
        self.backbone_name = backbone_name
        self.backbone = None
        if backbone_name == 'darknet':
            self.backbone = DarkNet()
        elif backbone_name == 'resnet':
            self.backbone = yolo_resnet(pretrained=pretrain)
        self.conv = None

        assert self.backbone is not None, 'backbone type not support'

        self.fc = nn.Sequential(
            nn.Linear(1024*s*s, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, s*s*cell_out_ch)
        )

    def forward(self, x):
        # if x is a picture, batch_size == 1
        batch_size = x.size(0)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # shift the dim, -1 means auto shift
        x = x.view(batch_size, self.s**2, -1)
        return x


class yolo_loss:
    def __init__(self, device, s, b, image_size, num_classes):
        self.device = device
        self.s = s
        self.b = b
        self.image_size = image_size
        self.num_classes = num_classes

    def __call__(self, input, target):
        """
        :param input: tensor[s, s, b*5 + n_class]  b: b * (center_x, center_y, w, h, obj_conf), class1_p, class2_p..
        :param target: (dataset type) tensor[n_bbox] bbox: (x_min, ymin, xmax, ymax, class)
        :return: loss tensor

        grid type: tensor[[bbox, ..], [], ..] -> bbox: (c_x, c_y, w, h, class)
        
        """
        self.batch_size = input.size(0)

        # label preprocess
        target = [self.label_direct2grid(target[i]) for i in range(self.batch_size)]

        # IOU match predictor and label
        # x, y, w, h, c
        match = []
        conf = []
        for i in range(self.batch_size):
            m, c = self.match_pred_target(input[i], target[i])
            match.append(m)
            conf.append(c)
        
        loss = torch.zeros([self.batch_size], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)
        for i in range(self.batch_size):
            loss[i], xy_loss[i], wh_loss[i], conf_loss[i], class_loss[i] = \
                self.compute_loss(input[i], target[i], match[i], conf[i])
        return torch.mean(loss), torch.mean(xy_loss), torch.mean(wh_loss), torch.mean(conf_loss), torch.mean(class_loss)

    def label_direct2grid(self, label):
        """
        :param label: dataset type: (xmin, ymin, xmax, ymax, class)
        :return: label: grid type: if this grid has object -> (center_x, center_y, w, h, conf)
                                   if this grid doesnot has object -> None
        """
        output = [None for _ in range(self.s ** 2)]
        size = self.image_size // self.s # get the size of grid

        n_bbox = label.size(0)
        label_grid = torch.zeros_like(label)

        label_grid[:, 0] = (label[:, 0] + label[:, 2]) / 2
        label_grid[:, 1] = (label[:, 1] + label[:, 3]) / 2
        label_grid[:, 2] = abs(label[:, 0] - label[:, 2])
        label_grid[:, 3] = abs(label[:, 1] - label[:, 3])
        label_grid[:, 4] = label[:, 4]

        idx_x = [int(label_grid[i][0]) // size for i in range(n_bbox)]
        idx_y = [int(label_grid[i][1]) // size for i in range(n_bbox)]

        label_grid[:, 0] = torch.div(torch.fmod(label_grid[:, 0], size), size)
        label_grid[:, 1] = torch.div(torch.fmod(label_grid[:, 1], size), size)
        label_grid[:, 2] = torch.div(label_grid[:, 2], self.image_size)
        label_grid[:, 3] = torch.div(label_grid[:, 3], self.image_size)

        for i in range(n_bbox):
            idx = idx_y[i] * self.s + idx_x[i]
            if output[idx] is None:
                output[idx] = torch.unsqueeze(label_grid[i], dim=0)
            else:
                output[idx] = torch.cat([output[idx], torch.unsqueeze(label_grid[i], dim=0)], dim=0)

        return output

    def match_pred_target(self, input, target):
        match = []
        conf = []
        with torch.no_grad():
            input_bbox = input[:, :self.b*5].reshape(-1, self.b, 5)
            ious = [match_get_iou(input_bbox[i], target[i], self.s, i) for i in range(self.s ** 2)]
            for iou in ious:
                if iou is None:
                    match.append(None)
                    conf.append(None)
                else:
                    keep = np.ones([len(iou[0])], dtype=bool)
                    m = []
                    c = []
                    for i in range(self.b):
                        if np.any(keep) == False:
                            break
                        # select the highest iou label
                        idx = np.argmax(iou[i][keep])
                        np_max = np.max(iou[i][keep])
                        m.append(np.argwhere(iou[i] == np_max).tolist()[0][0])
                        c.append(np.max(iou[i][keep]))
                        keep[idx] = 0
                    match.append(m)
                    conf.append(c)
        return match, conf

    def compute_loss(self, input, target, match, conf):
        ce_loss = nn.CrossEntropyLoss()

        input_bbox = input[:, :self.b*5].reshape(-1, self.b, 5)
        input_class = input[:, self.b*5:].reshape(-1, self.num_classes)

        input_bbox = torch.sigmoid(input_bbox)
        loss = torch.zeros([self.s ** 2], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)

        for i in range(self.s ** 2):
            # l: tensor[xy_loss, wh_loss, conf_loss, class_loss]
            l = torch.zeros([4], dtype=torch.float, device=self.device)
            # neg
            if target[i] is None:
                lambda_noobj = 0.5
                obj_conf_target = torch.zeros([self.b], dtype=torch.float, device=self.device)
                l[2] = torch.sum(torch.mul(lambda_noobj, torch.pow(input_bbox[i, :, 4] - obj_conf_target, 2)))
            else:
                lambda_coord = 5
                l[0] = torch.mul(lambda_coord, torch.sum(torch.pow(input_bbox[i, :, 0] - target[i][match[i], 0], 2) +
                                                         torch.pow(input_bbox[i, :, 1] - target[i][match[i], 1], 2)))

                l[1] = torch.mul(lambda_coord, torch.sum(torch.pow(torch.sqrt(input_bbox[i, :, 2]) - 
                                                                   torch.sqrt(target[i][match[i], 2]), 2) + 
                                                         torch.pow(torch.sqrt(input_bbox[i, :, 3]) - 
                                                                   torch.sqrt(target[i][match[i], 3]), 2)))
                obj_conf_target = torch.tensor(conf[i], dtype=torch.float, device=self.device)
                l[2] = torch.sum(torch.pow(input_bbox[i, :, 4] - obj_conf_target, 2))
                l[3] = ce_loss(input_class[i].unsqueeze(dim=0).repeat(target[i].size(0), 1), 
                               target[i][:, 4].long())

            loss[i] = torch.sum(l)
            xy_loss[i] = torch.sum(l[0])
            wh_loss[i] = torch.sum(l[1])
            conf_loss[i] = torch.sum(l[2])
            class_loss[i] = torch.sum(l[3])
        return torch.sum(loss), torch.sum(xy_loss), torch.sum(wh_loss), torch.sum(conf_loss), torch.sum(class_loss)





def centerxcenterywh2xywh(bbox):
    """
    :param bbox: tensor[bbox, bbox, ...] bbox: (center_x, center_y, w, h, conf)
    :return: tensor[bbox, bbox, ...] bbox: (x, y, w, h, conf)
    """
    bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox

def match_get_iou(bbox1, bbox2, s, idx):
    """
    :param bbox1: [bbox, bbox, ..] bbox: (center_x, center_y, w, h, conf)
    :return: [int, int, ..]
    """
    if bbox1 is None or bbox2 is None:
        return None

    bbox1 = np.array(bbox1.cpu())
    bbox2 = np.array(bbox2.cpu())

    # center_x, center_y to percentage that relative to image
    bbox1[:, 0] = bbox1[:, 0] / s
    bbox1[:, 1] = bbox1[:, 1] / s
    bbox2[:, 0] = bbox2[:, 0] / s
    bbox2[:, 1] = bbox2[:, 1] / s

    grid_pos = [(j / s, i / s) for i in range(s) for j in range(s)]
    bbox1[:, 0] = bbox1[:, 0] + grid_pos[idx][0]
    bbox1[:, 1] = bbox1[:, 1] + grid_pos[idx][1]
    bbox2[:, 0] = bbox2[:, 0] + grid_pos[idx][0]
    bbox2[:, 1] = bbox2[:, 1] + grid_pos[idx][1]

    # use percentage to compute
    bbox1 = centerxcenterywh2xywh(bbox1)
    bbox2 = centerxcenterywh2xywh(bbox2)

    return get_iou(bbox1, bbox2)

def get_iou(bbox1, bbox2):
    """
    :param bbox1: tensor[bbox, bbox, ..] bbox: (xmin, ymin, xmax, ymax, conf)
    :param bbox2: tensor[bbox, bbox, ..] bbox: (xmin, ymin, xmax, ymax, conf)
    :return: area: shape(num of bbox1, num of bbox2)
    """
    s1 = abs(bbox1[:, 2] - bbox1[:, 0]) * abs(bbox1[:, 3] - bbox1[:, 1])
    s2 = abs(bbox2[:, 2] - bbox2[:, 0]) * abs(bbox2[:, 3] - bbox2[:, 1])

    ious = []
    for i in range(bbox1.shape[0]):
        xmin = np.maximum(bbox1[i, 0], bbox2[:, 0])
        ymin = np.maximum(bbox1[i, 1], bbox2[:, 1])
        xmax = np.minimum(bbox1[i, 2], bbox2[:, 2])
        ymax = np.minimum(bbox1[i, 3], bbox2[:, 3])

        in_w = np.maximum(xmax - xmin, 0)
        in_h = np.maximum(ymax - ymin, 0)

        in_s = in_w * in_h
        iou = in_s / (s1[i] + s2 - in_s)
        # one iou means the ious that one bbox1 compute with all of bbox2 
        ious.append(iou)
    ious = np.array(ious)

    return ious

def nms(bbox, conf_th, iou_th):
    bbox = np.array(bbox.cpu())
    bbox[:, 4] = bbox[:, 4] * bbox[:, 5]
    bbox = bbox[bbox[:, 4] > conf_th]
    order = np.argsort(-bbox[:, 4])
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        iou = get_iou(np.array([bbox[i]]), bbox[order[1:]])[0]
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return bbox[keep]


def output_process(output, image_size, s, b, conf_th, iou_th):
    """
    :return output list[], bbox: (xmin, ymin,xmax, ymax, obj_conf, classes_conf, classes)
    """
    batch_size = output.size(0)
    size = image_size // s

    output = torch.sigmoid(output)

    # return max and its index
    classes_conf, classes = torch.max(output[:, :, b*5:], dim=2)
    classes = classes.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    classes_conf = classes_conf.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    bbox = output[:, :, :b*5].reshape(batch_size, -1, b, 5)
    bbox = torch.cat([bbox, classes_conf, classes], dim=3)
    bbox[:, :, :, [0, 1]] = bbox[:, :, :, [0, 1]] * size
    bbox[:, :, :, [2, 3]] = bbox[:, :, :, [2, 3]] * image_size

    grid_pos = [(j * image_size // s, i * image_size //s) for i in range(s) for j in range(s)]

    def to_direct(bbox):
        for i in range(s ** 2):
            bbox[i, :, 0] = bbox[i, :, 0] + grid_pos[i][0]
            bbox[i, :, 1] = bbox[i, :, 1] + grid_pos[i][1]
        return bbox
    
    bbox_direct = torch.stack([to_direct(b) for b in bbox])
    bbox_direct = bbox_direct.reshape(batch_size, -1, 7) # 7 = 5 + 2

    bbox_direct[:, :, 0] = bbox_direct[:, :, 0] - bbox_direct[:, :, 2] / 2
    bbox_direct[:, :, 1] = bbox_direct[:, :, 1] - bbox_direct[:, :, 3] / 2
    bbox_direct[:, :, 2] = bbox_direct[:, :, 0] + bbox_direct[:, :, 2]
    bbox_direct[:, :, 3] = bbox_direct[:, :, 1] + bbox_direct[:, :, 3]

    bbox_direct[:, :, 0] = torch.maximum(bbox_direct[:, :, 0], torch.zeros(1))
    bbox_direct[:, :, 1] = torch.maximum(bbox_direct[:, :, 1], torch.zeros(1))
    bbox_direct[:, :, 2] = torch.minimum(bbox_direct[:, :, 2], torch.tensor([image_size]))
    bbox_direct[:, :, 3] = torch.minimum(bbox_direct[:, :, 3], torch.tensor([image_size]))

    bbox = [torch.tensor(nms(b, conf_th, iou_th)) for b in bbox_direct]
    bbox = torch.stack(bbox)
    return bbox




if __name__ == "__main__":
    x = torch.randn([1, 3, 448, 448])
    yolo_net = yolo(7, 2*5+20, 'resnet', pretrain=None)
    print(yolo_net)
    out = yolo_net(x)
    print(out)
    print(out.size()) # (1, 49, 49)

    s = 2
    b = 2
    image_size = 448  # h, w
    input = torch.tensor([[[0.45, 0.24, 0.22, 0.3, 0.35, 0.54, 0.66, 0.7, 0.8, 0.8, 0.17, 0.9],
                           [0.37, 0.25, 0.5, 0.3, 0.36, 0.14, 0.27, 0.26, 0.33, 0.36, 0.13, 0.9],
                           [0.12, 0.8, 0.26, 0.74, 0.8, 0.13, 0.83, 0.6, 0.75, 0.87, 0.75, 0.24],
                           [0.1, 0.27, 0.24, 0.37, 0.34, 0.15, 0.26, 0.27, 0.37, 0.34, 0.16, 0.93]]])
    target = [torch.tensor([[200, 200, 353, 300, 1],
                            [220, 230, 353, 300, 1],
                            [15, 330, 200, 400, 0],
                            [100, 50, 198, 223, 1],
                            [30, 60, 150, 240, 1]], dtype=torch.float)]

    yolo_loss_obj = yolo_loss('cpu', 2, 2, image_size, 2)
    loss = yolo_loss_obj(input, target)
    print(loss)