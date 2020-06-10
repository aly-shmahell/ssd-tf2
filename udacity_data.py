import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random
import pandas as pd
from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial


class UdacityDataset():

    def __init__(self, data_dir, config, default_boxes,
                 new_size, num_examples=-1, augmentation=['original']):
        super(UdacityDataset, self).__init__()
        self.data_dir      = os.path.join(data_dir)
        self.config        = config
        self.default_boxes = default_boxes
        self.new_size      = new_size
        print("debug")
        print(augmentation)
        print(set(augmentation))
        print(set(augmentation)|set(['original']))
        self.augmentation  = list(set(augmentation)|set(['original']))
        self.labels        = dict([(v, k) for k, v in enumerate(['car', 'truck', 
                                                                 'pedestrian', 
                                                                 'bicyclist', 
                                                                 'traffic light'])])
        self.ids           = {
            'all': [
                x 
                for x in map(lambda x: x[:-4], os.listdir(self.data_dir)) 
                if 'jpg' in x
            ][:num_examples]
        }
        self.ids['train'] = self.ids['all'][:int(len(self.ids['all']) * 0.75)]
        self.ids['val']   = self.ids['all'][int(len(self.ids['all'])  * 0.75):]

    def __len__(self):
        return len(self.ids['all'])

    def _get_image(self, index):
        return Image.open(
            os.path.join(self.data_dir, self.ids['all'][index])
        )
        return img

    def _get_annotation(self, index, hw):
        (h, w)   = hw
        boxes    = []
        labels   = []
        df       = pd.read_csv(self.config)
        for idx, row in df[df['frame']==self.ids['all'][index]].iterrows():
            name = row['class_id'].lower().strip()
            xmin = float(row['xmin']) / w
            ymin = float(row['ymin']) / h
            xmax = float(row['xmax']) / w
            ymax = float(row['ymax']) / h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.labels[name] + 1)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset='all'):
        for index in range(len(self.ids[subset])):
            filename = indices['all'][index]
            img      = self._get_image(index)
            (boxes, labels) = self._get_annotation(index, img.size)
            boxes           = tf.constant(boxes, dtype=tf.float32)
            labels          = tf.constant(labels, dtype=tf.int64)
            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'patch':
                img, boxes, labels = random_patching(img, boxes, labels)
            elif augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)
            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)
            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels
            )
            yield filename, img, gt_confs, gt_locs


def create_batch_generator(data_dir, config, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=['original']):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    udacity = UdacityDataset(data_dir, config, default_boxes,
                     new_size, num_examples, augmentation)
    info = {
        'labels': udacity.labels,
        'length': len(udacity)
    }
    if mode == 'train':
        train_gen     = partial(udacity.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        val_gen       = partial(udacity.generate, subset='val')
        val_dataset   = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset   = val_dataset.batch(batch_size)
        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset       = tf.data.Dataset.from_generator(
            udacity.generate, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        dataset       = dataset.batch(batch_size)
        return dataset.take(num_batches), info
