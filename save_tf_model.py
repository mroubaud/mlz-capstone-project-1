#import libraries 
import tensorflow as tf
from tensorflow import keras

###PUT THE PATH AND THE NAME OF THE FILE FOR THE BEST MODEL YOU GOT HERE
model = keras.models.load_model('models/v5/horse_vs_humans_v5_01_0.517.h5')

tf.saved_model.save(model, 'horses_vs_humans')