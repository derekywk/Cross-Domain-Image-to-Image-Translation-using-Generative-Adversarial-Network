
# for importing packages
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import cv2
import itertools
import sys
from collections import Counter
sys.path.append(r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/models/research")

print(tf.__version__)
print(tf.config.list_physical_devices())

# declarations
batch_size = 32
img_height = 747
img_width = 1024
data_directory_path = r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_all_images_sm"
train_annotations_path = r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_annotation_files/train_annotations.json"
cis_test_annotations_path = r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_annotation_files/cis_test_annotations.json"
cis_val_annotations_path = r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_annotation_files/cis_val_annotations.json"
trans_test_annotations_path = r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_annotation_files/trans_test_annotations.json"
trans_val_annotations_path = r"Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_annotation_files/trans_val_annotations.json"

"""# Dataset Preparation"""

def data_set_generator(data_directory_path, annotations_path, loadImage=True):
  # if loadImage is False, just to yield image path as string
  with open(annotations_path) as f:
    annotation_file = json.load(f)

  category_dict = {item['id']: item['name'] for item in annotation_file['categories']}
  
  for annotation in annotation_file['annotations']:
    # yield tuple of (image, label, boundary box)
    image_path = rf"{data_directory_path}/{annotation['image_id']}.jpg"
    if annotation['category_id'] == 30:
      # images without animals
      if loadImage:
        yield (cv2.imread(image_path), category_dict[annotation['category_id']], None)
      else:
        yield (image_path, category_dict[annotation['category_id']], None)
    else:
      if 'bbox' not in annotation: 
        # leave images without bbox
        continue
      else:
        bbox = annotation['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox = [a/2 for a in bbox]
        if loadImage:
          yield (cv2.imread(image_path), category_dict[annotation['category_id']], bbox)
        else:
          yield (image_path, category_dict[annotation['category_id']], bbox)

def print_dataset_stats(data_set_generator):
  counter = Counter(data[1] for data in data_set_generator)
  print(counter)
#
# training_set_generator = data_set_generator(data_directory_path, train_annotations_path, loadImage=False)
# cis_test_set_generator = data_set_generator(data_directory_path, cis_test_annotations_path, loadImage=False)
# cis_val_set_generator = data_set_generator(data_directory_path, cis_val_annotations_path, loadImage=False)
# trans_test_set_generator = data_set_generator(data_directory_path, trans_test_annotations_path, loadImage=False)
# trans_val_set_generator = data_set_generator(data_directory_path, trans_val_annotations_path, loadImage=False)
#
# print_dataset_stats(training_set_generator)
# print_dataset_stats(cis_test_set_generator)
# print_dataset_stats(cis_val_set_generator)
# print_dataset_stats(trans_test_set_generator)
# print_dataset_stats(trans_val_set_generator)
#
# # testing print
# trans_val_set_generator = data_set_generator(data_directory_path, trans_val_annotations_path, loadImage=True)
# next(trans_val_set_generator)

"""# TFRecords Generation - Tensorflow API"""
with open(train_annotations_path) as f:
  annotation = json.load(f)

category_dict = {item['id']: item['name'] for item in annotation['categories']}

from object_detection.utils import dataset_util

def create_tf_example(image_dir, example):
  height, width, annotations = example
  filename = f"{annotations[0]['image_id']}.jpg" # Filename of the image. Empty if image is not from file
  with open(image_dir+'/'+filename,'rb') as f:
    encoded_image_data = f.read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [(annotation["bbox"][0])/width/2 for annotation in annotations if "bbox" in annotation] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [(annotation["bbox"][0] + annotation["bbox"][2])/width/2 for annotation in annotations if "bbox" in annotation] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [(annotation["bbox"][1])/height/2 for annotation in annotations if "bbox" in annotation] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [(annotation["bbox"][1] + annotation["bbox"][3])/height/2 for annotation in annotations if "bbox" in annotation] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  xmins = [a if 0 <= a <= 1 else 1 if a > 1 else 0 for a in xmins]
  xmaxs = [a if 0 <= a <= 1 else 1 if a > 1 else 0 for a in xmaxs]
  ymins = [a if 0 <= a <= 1 else 1 if a > 1 else 0 for a in ymins]
  ymaxs = [a if 0 <= a <= 1 else 1 if a > 1 else 0 for a in ymaxs]

  classes_text = [category_dict[annotation["category_id"]].encode('utf-8') for annotation in annotations] # List of string class name of bounding box (1 per box)
  classes = [annotation["category_id"] for annotation in annotations] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(int(height)),
      'image/width': dataset_util.int64_feature(int(width)),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def generate_tf_records(annotations_path, output_filebase, image_dir = r'Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/data/eccv_18_all_images_sm'):
  import contextlib2
  from object_detection.dataset_tools import tf_record_creation_util
  with open(annotations_path) as f:
    annotation_file = json.load(f)

  num_samples = len(annotation_file['annotations'])
  annotation_dict = {image["id"]: (image["height"]/2, image["width"]/2, []) for image in annotation_file['images']} # key: image_id, value: height, width, list of annotation
  for annotation in annotation_file['annotations']:
    annotation_dict[annotation['image_id']][2].append(annotation)
  num_shards = (num_samples // 4096) + (1 if num_samples % 4096 else 0 )

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
    for index, tf_example in enumerate(tf_example for tf_example in (create_tf_example(image_dir, example) for example in annotation_dict.values()) if tf_example is not None):
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

generate_tf_records(train_annotations_path, output_filebase=r'Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/tfrecords/train_dataset.record')
generate_tf_records(cis_val_annotations_path, output_filebase=r'Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/tfrecords/cis_val_dataset.record')
generate_tf_records(cis_test_annotations_path, output_filebase=r'Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/tfrecords/cis_test_dataset.record')
generate_tf_records(trans_val_annotations_path, output_filebase=r'Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/tfrecords/trans_val_dataset.record')
generate_tf_records(trans_test_annotations_path, output_filebase=r'Z:/Source/Wizpresso Personal/Derek Yuen/mlp/project/tfrecords/trans_test_dataset.record')

"""# Tensorflow Object Detection API Training"""

# Commented out IPython magic to ensure Python compatibility.
# cd /content/drive/MyDrive/tfmodel/research
# protoc object_detection/protos/*.proto --python_out=.
# cp object_detection/packages/tf2/setup.py .
# python -m pip install --use-feature=2020-resolver .
# pip install opencv-python-headless==4.1.2.30

# cd /content/drive/MyDrive/tfmodel/research
# python object_detection/model_main_tf2.py /
#  --pipeline_config_path=/content/pipeline_v2.config /
#  --model_dir=/content/model /
#  --alsologtostderr

# python models/research/object_detection/model_main_tf2.py --pipeline_config_path=pipeline_v2_local.config --model_dir=training --alsologtostderr