import tensorflow as tf

def parse_tfrecord(record):
    features = tf.compat.v1.parse_single_example(record, features={
        'shape': tf.compat.v1.FixedLenFeature([3], tf.int64),
        'data': tf.compat.v1.FixedLenFeature([], tf.string)})
    data = tf.compat.v1.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def add_noise(x):

    (minval,maxval) = (0.0, 50.0)
    shape = tf.shape(x)
    std = tf.random.uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
    return x + tf.random.normal(shape, stddev=std)

def random_crop_noised_clean(x):
    resized_image = tf.image.resize(x, size=[256,256,3], method='bicubic')/ 255.0 - 0.5
    return (add_noise(resized_image), add_noise(resized_image))

def tf_dataset(tfrecord_path, minibatch_size=64, num_threads=2, buf_size=1000, buffer_mb=256 ):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='', buffer_size=buffer_mb<<20)
    #dataset = dataset.repeat()
    #dataset = dataset.prefetch(buf_size)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=num_threads)
    #dataset = dataset.shuffle(buffer_size=buf_size)
    dataset = dataset.map(lambda x: random_crop_noised_clean(x))
    dataset = dataset.batch(minibatch_size)
    #dataset = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return dataset