import tensorflow as tf
import sklearn.model_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import cv2
import scipy.ndimage
import skimage.filters
import skimage.transform
import nibabel as nib
import timeit
from tqdm.notebook import tqdm


path = '/content/drive/My Drive/'
data_path = path + 'data/'
#path = '../working/'



from google.colab import drive
drive.mount('/content/drive')

x_train = np.load( data_path + 'train_x.npy' ).astype( np.float32 )
y_train = np.load( data_path + 'train_y.npy' ).astype( np.float32 )

x_val = np.load( data_path + 'val_x.npy' ).astype( np.float32 )
y_val = np.load( data_path + 'val_y.npy' ).astype( np.float32 )

x_test = np.load( data_path + 'test_x.npy' ).astype( np.float32 )
y_test = np.load( data_path + 'test_y.npy' ).astype( np.float32 )

x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape


def my_generator( x_train, y_train, batch_size ):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #featurewise_std_normalization=True,
        rotation_range=5,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=True
    ).flow( x_train, x_train, batch_size, seed = 1 )
    
    mask_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #featurewise_std_normalization=True,
        rotation_range=5,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=True
    ).flow( y_train, y_train, batch_size, seed = 1 )
    
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch

image_batch, mask_batch = next(
    my_generator( x_train, y_train, 8 ) )


def threshold( y ):
    return y[ y > 0.5 ]

def dice_coef(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: dice_coeff -- A metric that accounts for precision and recall
                           on the scale from 0 - 1. The closer to 1, the
                           better.
    '''
    y_true_f = tf.keras.backend.flatten( y_true )
    y_pred_f = tf.keras.backend.flatten( y_pred )
    intersection = tf.keras.backend.sum( y_true_f * y_pred_f )
    smooth = 1.0
    return ( 2.0 * intersection + smooth ) / ( tf.keras.backend.sum( y_true_f ) + tf.keras.backend.sum( y_pred_f ) + smooth )

def dice_coef_loss( y_true, y_pred ):
    return 1. - dice_coef( y_true, y_pred )

def iou( y_true, y_pred ):
    y_pred = tf.keras.backend.cast(
        tf.keras.backend.greater( y_pred, .5 ),
        dtype='float32'
    )
    
    inter = tf.keras.backend.sum( 
        tf.keras.backend.sum( 
            tf.keras.backend.squeeze( y_true * y_pred, axis=3 ), 
            axis=2 ),
        axis=1 
    )
    union = tf.keras.backend.sum( 
        tf.keras.backend.sum( 
            tf.keras.backend.squeeze( y_true + y_pred, axis=3 ), 
            axis=2 
        ), axis=1 
    ) - inter
    return tf.keras.backend.mean( ( inter + tf.keras.backend.epsilon() ) / ( union + tf.keras.backend.epsilon() ) )

num_samples = x_train.shape[0]
batch_size = 8
# steps_per_epoch * batch_size = number_of_rows_in_train_data
steps_per_epoch = num_samples // batch_size
epochs = 500
lr = 0.01

def scheduler(epoch):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp( 0.1 * ( 10 - epoch ) )

"""# BCDU-net model

"""

def BCDU_net( input_size = (128,128,1) ):
    N = input_size[0]
    inputs = tf.keras.layers.Input(input_size) 
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = tf.keras.layers.Dropout(0.5)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = tf.keras.layers.Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = tf.keras.layers.Dropout(0.5)(conv4_2)
    # D3
    merge_dense = tf.keras.layers.concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = tf.keras.layers.Dropout(0.5)(conv4_3)

    up6 = tf.keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
    up6 = tf.keras.layers.BatchNormalization(axis=3)(up6)
    up6 = tf.keras.layers.Activation('relu')(up6)

    x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = tf.keras.layers.concatenate([x1,x2], axis = 1) 
    merge6 = tf.keras.layers.ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = tf.keras.layers.BatchNormalization(axis=3)(up7)
    up7 = tf.keras.layers.Activation('relu')(up7)

    x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = tf.keras.layers.concatenate([x1,x2], axis = 1) 
    merge7 = tf.keras.layers.ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = tf.keras.layers.BatchNormalization(axis=3)(up8)
    up8 = tf.keras.layers.Activation('relu')(up8)    

    x1 = tf.keras.layers.Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = tf.keras.layers.Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = tf.keras.layers.concatenate([x1,x2], axis = 1) 
    merge8 = tf.keras.layers.ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = tf.keras.Model( inputs, conv9)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4), 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy']
    )
    return model

"""# x-net model
[X-Net: Brain Stroke Lesion Segmentation Basedon Depthwise Separable Convolution andLong-range Dependencies]
"""

def fsm( x ):
    channel_num = x.shape[-1]
    res = x
    x = conv2d_bn_relu( x, filters = int( channel_num // 8 ), kernel_size = ( 3, 3 ) )

    ip = x
    ip_shape = tf.keras.backend.int_shape( ip )
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    rank = 4
    if intermediate_dim < 1:
        intermediate_dim = 1
    # theta path
    theta = tf.keras.layers.Conv2D(
        intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer = tf.keras.regularizers.l2(1e-5)
    )( ip )
    theta = tf.keras.layers.Reshape( ( -1, intermediate_dim ) )( theta )

    # phi path
    phi = tf.keras.layers.Conv2D(
        intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer = tf.keras.regularizers.l2( 1e-5 )
    )( ip )
    phi = tf.keras.layers.Reshape((-1, intermediate_dim))(phi)

    # dot
    print( theta )
    print( phi )
    
    f = tf.keras.layers.Dot( axes = 2 )([ theta, phi ])
    print( f )
    size = tf.keras.backend.int_shape( f )
    #size = (1, 1, 64 )
    
    # scale the values to make it size invariant
    #print( size )
    #print( size[ -1 ] )
    
    #f = tf.keras.layers.Lambda( lambda z: ( 1. / float( size[ -1 ] ) ) * z )( f )
    ############################
    ###########################
    ###########################
    # g path
    g = tf.keras.layers.Conv2D(
        intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(ip)
    g = tf.keras.layers.Reshape( ( -1, intermediate_dim ) )(g)

    # compute output path
    y = tf.keras.layers.Dot( axes=[2, 1] )( [f, g] )
    y = tf.keras.layers.Reshape((dim1, dim2, intermediate_dim))(y)
    y = tf.keras.layers.Conv2D(
        channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer = tf.keras.regularizers.l2( 1e-5 )
    )(y)
    y = tf.keras.layers.Add()( [ ip, y ] )

    x = y
    x = conv2d_bn_relu( x, filters = int( channel_num ), kernel_size = ( 3, 3 ) )
    print( x )

    x = tf.keras.layers.Add()( [ x, res ] )
    return x

def conv2d_bn_relu( inputs, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                    kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2( 1e-5 )
                  ):
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer
    )( inputs )
    x = tf.keras.layers.BatchNormalization()( x )
    x = tf.keras.layers.ReLU()( x )
    return x

def depth_conv_bn_relu( inputs, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                        initializer='he_normal', regularizer=tf.keras.regularizers.l2(1e-5)
                      ):
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding,
        depthwise_initializer=initializer, use_bias=False, depthwise_regularizer=regularizer
    )( inputs )
    x = tf.keras.layers.BatchNormalization()( x )
    x = tf.keras.layers.ReLU()( x )
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding=padding,
        kernel_initializer=initializer, kernel_regularizer=regularizer
    )( x )
    x = tf.keras.layers.BatchNormalization()( x )
    x = tf.keras.layers.ReLU()( x )
    return x

def x_block( x, channels ):
    res = conv2d_bn_relu( x, filters=channels, kernel_size=(1, 1) )
    x = depth_conv_bn_relu( x, filters=channels, kernel_size=(3, 3) )
    x = depth_conv_bn_relu( x, filters=channels, kernel_size=(3, 3) )
    x = depth_conv_bn_relu( x, filters=channels, kernel_size=(3, 3) )
    x = tf.keras.layers.Add()( [ x, res ] )
    return x

def create_xception_unet_n( input_shape = (128, 128, 1), pretrained_weights_file=None ):
    inputs = tf.keras.layers.Input(input_shape)

    # stage_1
    x = x_block( inputs, channels=64)
    stage_1 = x

    # stage_2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=128)
    stage_2 = x

    # stage_3
    x = tf.keras.layers.MaxPooling2D( pool_size = ( 2, 2 ) )( x )
    x = x_block( x, channels=256 )
    stage_3 = x

    # stage_4
    x = tf.keras.layers.MaxPooling2D( pool_size = ( 2, 2 ) )( x )
    x = x_block( x, channels=512 )
    stage_4 = x

    # stage_5
    x = tf.keras.layers.MaxPooling2D( pool_size = ( 2, 2 ) )( x )
    x = x_block( x, channels=1024 )
    
    # fsm
    x = fsm( x )

    # stage_4
    x = tf.keras.layers.UpSampling2D( size = ( 2, 2 ) )( x )
    x = conv2d_bn_relu( x, filters = 512, kernel_size = 3 )
    x = tf.keras.layers.Concatenate()( [ stage_4, x ] )
    x = x_block( x, channels = 512 )

    # stage_3
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = conv2d_bn_relu(x, filters=256, kernel_size=3)
    x = tf.keras.layers.Concatenate()([stage_3, x])
    x = x_block(x, channels=256)

    # stage_2
    x = tf.keras.layers.UpSampling2D( size = ( 2, 2 ) )( x )
    x = conv2d_bn_relu( x, filters=128, kernel_size = 3 )
    x = tf.keras.layers.Concatenate()( [ stage_2, x ] )
    x = x_block( x, channels = 128 )

    # stage_1
    x = tf.keras.layers.UpSampling2D( size = ( 2, 2 ) )( x )
    x = conv2d_bn_relu( x, filters = 64, kernel_size = 3 )
    x = tf.keras.layers.Concatenate()( [ stage_1, x ] )
    x = x_block( x, channels = 64 )

    # output
    x = tf.keras.layers.Conv2D( filters=1, kernel_size=1, activation='sigmoid' )( x )

    # create model
    model = tf.keras.Model( inputs=inputs, outputs=x)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),  # 2e-4 
        loss = dice_coef_loss,
        metrics = [
            #'accuracy', 
            iou,
            dice_coef 
        ]
    )

    print('Create X-Net with input shape = {}, output shape = {}'.format( input_shape, model.output.shape ) )

    # load weights
    if pretrained_weights_file is not None:
        model.load_weights( pretrained_weights_file, by_name=True, skip_mismatch=True )

    return model




#model = BCDU_net( input_size = ( 128, 128, 1 ) )
model = create_xception_unet_n( input_shape = (128, 128, 1), pretrained_weights_file = None )


tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = path + 'x-Net_model1.hdf5', 
        verbose = 0, 
        save_best_only = True,
        save_weights_only = False,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = path + 'x-Net_weights1.hdf5', 
        verbose = 0, 
        save_best_only = False,
        save_weights_only = True,
    ),
]

model.load_weights( path + 'x-Net_weights1.hdf5' )

history = model.fit(
    my_generator( x_train, y_train, batch_size ),
    steps_per_epoch = steps_per_epoch,
    validation_data = ( x_val, y_val ),
    epochs = epochs, 
    verbose = 1,
    callbacks = callbacks
)

_,_,_ = model.evaluate(  x_test, y_test, verbose = 1 )
_,_,_ = model.evaluate(  x_val, y_val, verbose = 1 )

model1 = tf.keras.models.load_model(
    path + 'x-Net_model.hdf5',
    custom_objects = {
        'dice_coef_loss': dice_coef_loss, 
        'dice_coef': dice_coef,
        'iou_iou': iou_loss,
        'iou': iou,
    }
)
_, _, _ = model1.evaluate(  x_test, y_test, verbose = 1 )

def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    
display_training_curves(
    history.history['loss'], 
    history.history['val_loss'], 
    'loss', 
    211
)
display_training_curves(
    history.history['dice_coef'], 
    history.history['val_dice_coef'], 
    'dice coef', 
    212
)
