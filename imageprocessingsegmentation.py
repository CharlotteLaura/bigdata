#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import sys
import itertools
import math
import logging
import json
import re
import random
import time
import concurrent.futures
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
from imgaug import augmenters as iaa


# In[15]:


# Root directory of the project
os.chdir('/Users/charlottemarks/Desktop/Mask_RCNN/samples/lung')
ROOT_DIR = os.getcwd()


# In[16]:


print(ROOT_DIR)


# In[17]:


if ROOT_DIR.endswith("samples/lung"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))


# In[18]:


print(ROOT_DIR)


# In[19]:


sys.path.append(ROOT_DIR)
from mrcnn import utils

from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log


# In[20]:


import lung

get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


ROOT_DIR


# In[29]:


DATASET_DIR = os.path.join(ROOT_DIR, "datasets/lung")

# Use configuation from lung.py, but override
# image resizing so we see the real sizes here
class NoResizeConfig(lung.LungConfig):
    IMAGE_RESIZE_MODE = "none"
    
config = NoResizeConfig()


# In[30]:


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# In[31]:


DATASET_DIR


# In[32]:



# Load dataset
dataset = lung.LungDataset()
# The subset is the name of the sub-directory, such as stage1_train,
# stage1_test, ...etc. You can also use these special values:
#     train: loads stage1_train but excludes validation images
#     val: loads validation images from stage1_train. For a list
#          of validation images see lung.py
dataset.load_lung(DATASET_DIR, subset="train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# In[34]:


# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)


# In[37]:


# Example of loading a specific image by its source ID
#source_id = "ed5be4b63e9506ad64660dd92a098ffcc0325195298c13c815a73773f1efc279"
source_id = "ID_0035_Z_0224"
# Map source ID to Dataset image_id
# Notice the lung prefix: it's the name given to the dataset in LungDataset
image_id = dataset.image_from_source_map["lung.{}".format(source_id)]

# Load and display
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id, use_mini_mask=False)
log("molded_image", image)
log("mask", mask)
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
                            #,show_bbox=False)


# In[ ]:


dataset.image_ids


# In[40]:


def image_stats(image_id):
    """Returns a dict of stats for one image."""
    image = dataset.load_image(image_id)
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id, use_mini_mask=False)
    log("molded_image", image)
    log("mask", mask)
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
                            #,show_bbox=False)
#     mask, _ = dataset.load_mask(image_id)
#     bbox = utils.extract_bboxes(mask)
#     # Sanity check
#     assert mask.shape[:2] == image.shape[:2]
#     # Return stats dict
#     return {
#         "id": image_id,
#         "shape": list(image.shape),
#         "bbox": [[b[2] - b[0], b[3] - b[1]]
#                  for b in bbox
#                  # Uncomment to exclude nuclei with 1 pixel width
#                  # or height (often on edges)
#                  # if b[2] - b[0] > 1 and b[3] - b[1] > 1
#                 ],
#         "color": np.mean(image, axis=(0, 1)),
#    }

# Loop through the dataset and compute stats over multiple threads
# This might take a few minutes
t_start = time.time()
#with concurrent.futures.ThreadPoolExecutor() as e:
with concurrent.futures.ProcessPoolExecutor() as e:
    list(e.map(image_stats, dataset.image_ids))
    #stats = list(e.map(image_stats, dataset.image_ids))
t_total = time.time() - t_start
print("Total time: {:.1f} seconds".format(t_total))


# In[42]:


image = dataset.load_image(image_id)


# In[43]:


image


# In[39]:


t_start = time.time()
for image_id in dataset.image_ids:
#def image_stats(image_id):
    """Returns a dict of stats for one image."""
    image = dataset.load_image(image_id)
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id, use_mini_mask=False)
    log("molded_image", image)
    log("mask", mask)
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
                            #,show_bbox=False)
#     mask, _ = dataset.load_mask(image_id)
#     bbox = utils.extract_bboxes(mask)
#     # Sanity check
#     assert mask.shape[:2] == image.shape[:2]
#     # Return stats dict
#     return {
#         "id": image_id,
#         "shape": list(image.shape),
#         "bbox": [[b[2] - b[0], b[3] - b[1]]
#                  for b in bbox
#                  # Uncomment to exclude nuclei with 1 pixel width
#                  # or height (often on edges)
#                  # if b[2] - b[0] > 1 and b[3] - b[1] > 1
#                 ],
#         "color": np.mean(image, axis=(0, 1)),
#    }

# Loop through the dataset and compute stats over multiple threads
# This might take a few minutes
#t_start = time.time()
#with concurrent.futures.ThreadPoolExecutor() as e:
# with concurrent.futures.ProcessPoolExecutor() as e:
#     list(e.map(image_stats, dataset.image_ids))
    #stats = list(e.map(image_stats, dataset.image_ids))
t_total = time.time() - t_start
print("Total time: {:.1f} seconds".format(t_total))
