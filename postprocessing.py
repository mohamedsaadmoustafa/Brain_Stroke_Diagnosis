import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2
import timeit
from tqdm.notebook import tqdm

# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/My Drive/'
#path = '../working/'

x_test = np.load( path + 'X_test.npy' ).astype( np.float32 )
y_test = np.load( path + 'Y_test.npy' ).astype( np.float32 )



def my_generator( x_train, y_train, batch_size ):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #featurewise_std_normalization=True,
        rotation_range = 5,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip = True
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
    my_generator( x_test, y_test, 8 ) )
    
'''
fig, ax = plt.subplots( 8, 4, figsize = ( 8, 20 ) )
for i in range(8):
    ax[i, 0].imshow( image_batch[ i, :, :, 0 ] )

    ax[i, 1].imshow( mask_batch[ i, :, :, 0 ] )

    ax[i, 2].imshow( np.ma.masked_where( mask_batch[ i, :, :, 0 ] == 1, image_batch[ i, :, :, 0 ] ) );

    ax[i, 3].imshow( image_batch[ i, :, :, 0  ] * 255. );
    ax[i, 3].imshow( np.ma.masked_where( mask_batch[ i, :, :, 0 ] == 1, mask_batch[ i, :, :, 0 ] ) * 255., alpha=0.5 );


    ax[i, 0].set_title( 'MRI Slice' )
    ax[i, 1].set_title( 'Mask' )
    ax[i, 2].set_title( 'white' )
    ax[i, 3].set_title( 'lesion area' )
'''

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

"""# x-net model
[X-Net: Brain Stroke Lesion Segmentation Basedon Depthwise Separable Convolution andLong-range Dependencies]
"""

model = tf.keras.models.load_model(
    path + 'x-Net_model-best.hdf5',
    custom_objects = {
        'dice_coef_loss': dice_coef_loss, 
        'dice_coef': dice_coef
    }
)

Dice_Coef_Loss, accuracy, iou, Dice_Coef = model.evaluate(  x_test, y_test, verbose = 1 )

print( f"Dice_Coef_Loss {Dice_Coef_Loss}, accuracy {accuracy}, iou {iou/2}, Dice_Coef {Dice_Coef}" )

"""# Plot results

- Threshold lesion at 0.5
- Dilate lesion
"""

t = 100 #x_test.shape[0]

image_batch, mask_batch = next(
    my_generator( x_test, y_test, t ) )

pred_batch = model.predict( image_batch )
print( pred_batch.shape )

image_batch = image_batch.squeeze()
mask_batch = mask_batch.squeeze()
pred_batch = pred_batch.squeeze()

fig, ax = plt.subplots( t, 7, figsize = (  15, 320 ) )

for i in tqdm( range( t ) ):
    ax[i, 0].imshow( image_batch[ i ] * 255. );
    ax[i, 1].imshow( mask_batch[ i ] * 255. );
    ax[i, 2].imshow( np.ma.masked_where( mask_batch[i] == 1, image_batch[i] ) * 255. );
    ax[i, 3].imshow( image_batch[ i ] * 255. );
    ax[i, 3].imshow( np.ma.masked_where( mask_batch[ i ] == 1, mask_batch[ i ] ) * 255., alpha=0.5 );

    # Predicted Area
    ax[i, 4].imshow( pred_batch[ i ] * 255. );

    # Threshold predictions
    pred_threshold  = ( pred_batch[i] > 0.5 ).astype( np.uint8 )
    # show masked image
    kernel = np.ones( ( 5, 5 ), np.uint8 ) ############## change kernel shape
    pred_threshold = cv2.dilate( pred_threshold * 255., kernel, iterations = 1 )
    
    ax[i, 5].imshow( pred_threshold );

    ax[i, 6].imshow( image_batch[ i ] * 255. );
    ax[i, 6].imshow( np.ma.masked_where( pred_threshold == 1, pred_threshold ) * 255., alpha=0.5 );


    # set titles
    ax[i, 0].set_title( 'MRI Slice' )
    ax[i, 1].set_title( 'Mask' )
    ax[i, 2].set_title( 'white' )
    ax[i, 3].set_title( 'lesion area' )
    
    ax[i, 4].set_title( 'Predicted' )
    ax[i, 5].set_title( 'Dialate Predicted' )
    #ax[i, 6].set_title( 'white' )
    ax[i, 6].set_title( 'Predicted lesion area' )

"""# Detect Segmented Area"""

img = image_batch[ 4 ]
label = mask_batch[ 4 ]
pred = pred_batch[ 4 ]

img.shape

np.unique( pred )

# Threshold predictions
pred_threshold  = ( pred > 0.5 ).astype( np.uint8 )

np.unique( pred_threshold )

img_masked = np.ma.masked_where( pred_threshold == 1, img )

img_masked2 = cv2.bitwise_and( img, img, mask=pred_threshold )

OPACITY = 0.7

#img_masked3 = cv2.addWeighted( img, OPACITY, pred_threshold, 1-OPACITY, 0 )

# dialate expected lesion area
#kernel = cv.getStructuringElement( cv.MORPH_CROSS,( 5, 5 ) )
kernel = np.ones( ( 5, 5 ), np.uint8 )
pred_threshold = cv2.dilate( pred_threshold * 255., kernel, iterations = 1 )

fig, ax = plt.subplots( 1, 7, figsize = ( 18, 80 ) )

ax[0].imshow( img * 255. );
ax[1].imshow( label * 255. );

ax[2].imshow( img * 255. );
ax[2].imshow( np.ma.masked_where( label == 1, label ) * 255., alpha=0.5 );

ax[3].imshow( pred * 255. );
ax[4].imshow( pred_threshold * 255. );
ax[5].imshow( img_masked * 255. );
ax[6].imshow( img_masked2 * 255. );
#ax[7].imshow( img_maske3 * 255. );

