import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, Concatenate, UpSampling2D
from tensorflow.keras import models, utils

def ConvBlock(filters, kernel_size, name):
    def wrapper(input_tensor):
        x = Conv2D(filters, kernel_size, padding='same', name = 'conv2d_' + name)(input_tensor)
        x = BatchNormalization(name = 'batch_normalization_' + name)(x)
        x = ReLU(name = 'relu_' + name)(x)
        return x
    return wrapper

def DeconvBlock():

    def wrapper(input_tensor, filters, skip, step):

        x = UpSampling2D((2, 2), name = "decoder_up" + str(step))(input_tensor)
        x = Concatenate(axis = 3, name='decoder_concat_' + str(step))([x, skip])
        x = ConvBlock(filters, 3, name = "decoder_conv_" + str(step)+"_a")(x)
        x = ConvBlock(filters, 3, name = "decoder_conv_" + str(step)+"_b")(x)

        return x

    return wrapper

def UNet():
    input_ = tf.keras.Input(shape=(256,256,3))

    x = ConvBlock(16, 3, '1a')(input_)
    x = ConvBlock(16, 3, '1b')(x)
    x = MaxPool2D(name = "max_pool_1")(x)
    
    x = ConvBlock(32, 3, '2a')(x)
    x = ConvBlock(32, 3, '2b')(x)
    x = MaxPool2D(name = "max_pool_2")(x)
    
    x = ConvBlock(64, 3, '3a')(x)
    x = ConvBlock(64, 3, '3b')(x)
    x = MaxPool2D(name = "max_pool_3")(x)
    
    x = ConvBlock(128, 3, '4a')(x)
    x = ConvBlock(128, 3, '4b')(x)
    x = MaxPool2D(name = "max_pool_4")(x)
    
    x = ConvBlock(256, 3, '5a')(x)
    x = ConvBlock(256, 3, '5b')(x)

    x.shape

    model = models.Model(input_, x)

    skip_connection_layers = ['relu_4b','relu_3b', 'relu_2b', 'relu_1b']
    skips = ([model.get_layer(name=i).output for i in skip_connection_layers])
    
    filters = [128, 64, 32, 16]
    
    input_ = model.input
    x = model.output
    
    for i in range(len(skip_connection_layers)):
        skip = skips[i]        
        x = DeconvBlock()(x, filters = filters[i], skip=skip, step = i)
        
    x = Conv2D(3, 3, padding='same', name='final')(x)

    unet_model = models.Model(input_, x)

    unet_model.summary()

    dot_img_file = 'model_1.png'
    utils.plot_model(unet_model, to_file=dot_img_file, show_shapes=True)

    return unet_model