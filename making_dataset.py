import tensorflow as tf

data_path = 'dataset\dataset.tfrecords'
buffer_mb = 256
minibatch_size = 64
buf_size = 1000
num_threads = 2

def parse_features(feature):
    features = tf.io.parse_single_example(feature, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    return tf.reshape(image, features['shape'])    

def parse_tfrecord_tf(record):
    features = tf.compat.v1.parse_single_example(record, features={
        'shape': tf.compat.v1.FixedLenFeature([3], tf.int64),
        'data': tf.compat.v1.FixedLenFeature([], tf.string)})
    data = tf.compat.v1.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])


def resize_small_image(x):
    shape = tf.shape(x)
    return tf.cond(
        tf.logical_or(
            tf.less(shape[2], 256),
            tf.less(shape[1], 256)
        ),
        true_fn=lambda: tf.image.resize(x, size=[256,256], method='bicubic'),
        false_fn=lambda: tf.cast(x, tf.float32)
     )

def add_train_noise_tf(x):
    train_stddev_range=(0.0, 50.0)

    (minval,maxval) = train_stddev_range
    shape = tf.shape(x)
    rng_stddev = tf.random.uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
    return x + tf.random.normal(shape) * rng_stddev
    
def random_crop_noised_clean(x, add_noise):
    cropped = tf.image.random_crop(resize_small_image(x), size=[256, 256, 3])/ 255.0 - 0.5
    return (add_noise(cropped), add_noise(cropped), cropped)


dataset = tf.data.TFRecordDataset('dataset\dataset.tfrecords', compression_type='', buffer_size=buffer_mb<<20)
dataset = dataset.repeat()
dataset = dataset.prefetch(buf_size)
dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
dataset = dataset.shuffle(buffer_size=buf_size)
dataset = dataset.map(lambda x: random_crop_noised_clean(x, add_train_noise_tf))
dataset = dataset.batch(minibatch_size)
dataset = tf.compat.v1.data.make_one_shot_iterator(dataset)

import cv2
import numpy as np
 
img = cv2.imread('dataset/train_mask/p1_2_001_mask.png', cv2.IMREAD_GRAYSCALE)/255 -0.44
 
gauss = np.random.normal(0,1,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
noise = img + img * gauss
 
cv2.imshow('a',noise)
cv2.waitKey(0)