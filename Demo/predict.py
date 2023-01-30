import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os



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


model = tf.keras.models.load_model(
	'models/x-Net_model-best.hdf5',
	custom_objects = {
		'dice_coef_loss': dice_coef_loss, 
		'dice_coef': dice_coef
	}
)
	
def predict_f( x, model=model ):

	x = np.load( x, allow_pickle=True )
	x = np.expand_dims( x, axis=0 )

	print( 'image shape: ', x.shape )

	y = model.predict( x )

	x = x[0, :, :, 0]
	y = y[0, :, :, 0]

	print( 'predicted' )

	return x, y