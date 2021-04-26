from making_dataset import tf_dataset
from network import UNet
import os

data_path = 'dataset\dataset.tfrecords'
buffer_mb = 256
minibatch_size = 64
buf_size = 1000
num_threads = 2
no_of_iteration = 20

DATASET_SIZE = len([img for img in os.listdir('dataset/images')])

dataset = tf_dataset(data_path)

train_size = int(0.9 * DATASET_SIZE)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

unet = UNet()

unet.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fiiting our NN to train set
r = unet.fit(train_dataset, val_dataset, epochs = no_of_iteration)