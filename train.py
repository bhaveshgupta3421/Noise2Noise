from making_dataset import tf_dataset
from network import UNet

data_path = 'dataset\dataset.tfrecords'
buffer_mb = 256
minibatch_size = 64
buf_size = 1000
num_threads = 2

dataset = tf_dataset(data_path)

unet = UNet()

unet.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fiiting our NN to train set
r = unet.fit(dataset, epochs = 2)
