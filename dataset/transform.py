import torch
import torchvision
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class ToTensor:
    def __init__(self):
        self.totensor = torchvision.transforms.ToTensor()
    
    def __call__(self, image, label):
        image = self.totensor(image)
        label = torch.tensor(label)
        return image, label


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

        """
        lable: (xmin, ymin, xmax, ymax)
        """
    def __call__(self, image, label):
        if random.random() < self.p:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = label[:, :4]
            bbox[:, [0, 2]] = width - bbox[:, [0, 2]]
            label[:, :4] = bbox
        return image, label


class Resize:
    def __init__(self, image_size, keep_ratio=True):
        """
        image_size: int
        keep_ratio=True: keep the origin ratio of h and w
        keep_ratio=False: fill into square 
        """
        self.image_size = image_size
        self.keep_ratio = keep_ratio

    def __call__(self, image, label):
        """
        image: tensor[3, h, w]
        label: (xmin, ymin, xmax, ymax)
        """
        h, w = tuple(image.size()[1:])
        label[:, [0, 2]] = label[:, [0, 2]] / w
        label[:, [1, 3]] = label[:, [1, 3]] / h
        
        if self.keep_ratio:
            r_h = min(self.image_size / h, self.image_size / w)
            r_w = r_h
        else:
            r_h = self.image_size / h
            r_w = self.image_size / w

        h, w = int(r_h * h), int(r_w * w)
        h, w = min(h, self.image_size), min(w, self.image_size)
        label[:, [0, 2]] = label[:, [0, 2]] * w
        label[:, [1, 3]] = label[:, [1, 3]] * h

        T = torchvision.transforms.Resize([h, w])

        # padding on the right and bottom
        Padding = torch.nn.ZeroPad2d((0, self.image_size-w, 0, self.image_size-h))
        image = Padding(T(image))

        assert list(image.size()) == [3, self.image_size, self.image_size]

        return image, label