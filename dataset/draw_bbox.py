import numpy as np
import torchvision.transforms as F
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

colors = ['Pink', 'Crimson', 'Magenta', 'Indigo', 'BlueViolet',
          'Blue', 'GhostWhite', 'LightSteelBlue', 'Brown', 'SkyBlue',
          'Tomato', 'SpringGreen', 'Green', 'Yellow', 'Olive',
          'Gold', 'Wheat', 'Orange', 'Gray', 'Red']

def draw(image, bbox, classes, show_conf=False, conf_th=0.0):
    keys = list(classes.keys())
    values = list(classes.values())
    # font = ImageFont.truetype('arial.ttf', 10)
    transform = F.ToPILImage()
    image = transform(image)
    draw_image = ImageDraw.Draw(image)

    bbox = np.array(bbox.cpu())

    for b in bbox:
        if show_conf and b[-2] < conf_th:
            continue
        draw_image.rectangle(list(b[:4]), outline=colors[int(b[-1])], width=3)
        if show_conf:
            draw_image.text(list(b[:2] + 5), keys[values.index(int(b[-1]))] + ' {:.2f}'.format(b[-2]),
                            fill=colors[int(b[-1])])
        else:
            draw_image.text(list(b[:2] + 5), keys[values.index(int(b[-1]))],
                            fill=colors[int(b[-1])])
    
    plt.figure()
    plt.imshow(image)
    plt.show()
    