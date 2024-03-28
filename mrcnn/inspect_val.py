import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
import re
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.model import mold_image
from mrcnn.visualize import random_colors
from mrcnn.visualize import apply_mask
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from itertools import groupby


# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import hgg 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

custom_WEIGHTS_PATH = "logs/hgg20240327T2037/mask_rcnn_hgg_0030.h5"  # TODO: update this path
config = hgg.hggConfig()
custom_DIR = os.path.join(ROOT_DIR, "data")
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = pyplot.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    colors = colors or random_colors(N)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    pyplot.savefig('mask_hgg.png')


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Load validation dataset
dataset = hgg.hggDataset()
dataset.load_hgg(custom_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
# load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

imgs=list()
# Load image
for image_id in dataset.image_ids:
#image_id = random.choice(dataset.image_ids)
#    if image_id>3:
#        continue
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))
    #print(dataset.image_reference(image_id)[42:])

    # Run object detection
    results = model.detect([image], verbose=1)

    numbers = re.findall(r'\d+', dataset.image_reference(image_id)[42:])
    img={"image":image, "ref":numbers[0], "slice_num":numbers[1], "true_diagnosis":numbers[2], "results":results[0]}
    imgs.append(img)
    # Display results
    ax = get_ax(1)
    r = results[0]
    """
    display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    """
imgs=sorted(imgs, key=lambda x: x['ref'])
grouped_imgs = {}
for key, group in groupby(imgs, key=lambda x: int(x['ref'])):
    grouped_imgs[key] = list(group)

corr=0
wrong=0
tolerance=3
for key, value in grouped_imgs.items():
    #print(key)

    success=0
    perc_suc=0.0
    suc_count=0
    perc_unsuc=0.5 
    last_slice=-1
    true_diagnosis=int(value[0]["true_diagnosis"])

    value=sorted(value, key=lambda x: int(x['slice_num']))

    for i in range(len(list(value))):
        img=value[i]
        curr_slice=int(img["slice_num"])

        scores=img["results"]["scores"];
        #print(img["ref"],img["slice_num"],scores)

        if len(list(scores))==0:
            perc_unsuc+=(1-perc_unsuc)/2
        else:
            if last_slice==-1:
                last_slice=curr_slice

            #print(curr_slice, last_slice)
            if curr_slice-last_slice<=tolerance:
                suc_count+=1;
                last_slice=curr_slice

            for j in range(len(list(scores))):
                if scores[j]>perc_suc:
                    perc_suc=scores[j]

        if suc_count>=3:
            success=1

    if success==1:
        print(true_diagnosis,'pred: 1',perc_suc)
    else:
        print(true_diagnosis,'pred: 0',perc_unsuc)

    if success==true_diagnosis:
        corr+=1
    else:
        wrong+=1
        
print(int(corr), int(wrong))
print(corr/(corr+wrong))

#log("gt_class_id", gt_class_id)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)
