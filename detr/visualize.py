import torch
from matplotlib import pyplot as plt
import os
import numpy as np
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

def plot_image(ax, img, norm):
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.astype('uint8')
    ax.imshow(img)


def plot_results(pil_img, prob, boxes, output_dir, classes, targets, ood=False):
    plt.figure(figsize=(16, 10))
    # plt.imshow(pil_img)
    ax = plt.gca()
    image = plot_image(ax, pil_img, True)
    # breakpoint()
    for p, cl, (xmin, ymin, xmax, ymax), c in zip(prob, classes, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        # cl = p.argmax()
        text = f'{CLASSES[cl]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # plt.show()
    print('hhh')

    if ood:
        plt.savefig(os.path.join(output_dir + '/images_ood', f'img_{int(targets[0]["image_id"][0])}.jpg'))
    else:
        plt.savefig(os.path.join(output_dir + '/images', f'img_{int(targets[0]["image_id"][0])}.jpg'))


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox)
    # breakpoint()
    return b

def visualize_prediction_results(samples, result, output_dir, targets, ood):
    # breakpoint()
    probas = result[0]['scores']
    keep = probas > 0.5
    # breakpoint()
    images = samples.tensors[0].cpu().permute(1,2,0).numpy()
    # breakpoint()
    bboxes_scaled = rescale_bboxes(result[0]['original_boxes'], list(images.shape[:2])[::-1])[keep]
    # bboxes_scaled = result[0]['boxes'][keep]

    classes = result[0]['labels'][keep]
    plot_results(images, probas[keep], bboxes_scaled,
                 output_dir, classes, targets, ood)
    # breakpoint()
    return