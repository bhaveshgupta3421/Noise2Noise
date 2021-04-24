import os
import tensorflow as tf
import cv2
import numpy as np

dir_path = 'dataset\\images'

images_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]

def _int64_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert len(im.shape) == 3
    im = im.astype(np.uint8)
    return im

writer = tf.io.TFRecordWriter('dataset\dataset.tfrecords')
    
for (idx, path) in enumerate(images_path[:100]):
    print (idx, path)
    image = load_image(path)
    feature = {
        'shape': _int64_feature(image.shape),
        'data': _byte_feature(tf.compat.as_bytes(image.tobytes()))
        }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())