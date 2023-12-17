#import libraries 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#dataset download and general info
#ds, info = tfds.load('horses_or_humans', with_info=True)
#builder = tfds.builder('horses_or_humans')

#Preprocessing Images
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

#train dataset
train_ds = train_datagen.flow_from_directory("./datasets/train/",
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
#validation dataset
val_ds = val_datagen.flow_from_directory("./datasets/test/",
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')



#Trabajo ahora con imagenes de 150x150 y Agrego check point para encontrar mejor epoch
#función que toma learning_rate, size_inner y droprate y devuelve modelo entrenado
def make_model(input_size=150, learning_rate=0.001, size_inner=100, droprate=0.2):
    
    ###############################################################
    ###---------MODEL STRUCTURE----------###
    
    ##---CONV LAYER---##
    #Create initial model
    model = models.Sequential()
    #Create conv layer of 
    #32 filters, which means 32 outputs
    #stride of size (3,3). Stride is the matrix output dim of each of the 32 filters
    #Reulu activation function
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(input_size, input_size, 3)))
    #MaxPooling2D((2, 2) transfor filter output to reduce size in 1/4
    #max pooling layer with a pooling window of size (2,2) will take a 2x2 section of the image, 
    #keep the max value and discard the rest. 
    #This layer will eventually output a feature map whose sides have been halved in size.
    model.add(layers.MaxPooling2D((2, 2)))
    #Get vector representation with Flatten
    model.add(layers.Flatten())

    
    ##---DENSE LAYER---##
    #Apply Dense layers (total of size_inner) to vector representation
    model.add(layers.Dense(size_inner, activation='relu'))
    
    ##---DROPOUT---##
    #el drop_rate to randomly eliminate parts of the image in order to avoid the model learns particular things
    #and not general things of the image
    model.add(layers.Dropout(droprate))
    
    ##---FINAL OUTPUT---##
    #Get final output of the binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    ###-----DENSE LAYER PARAMETERS-----###
    #this is what we use to train our model and calculare the weights
    #Adam use gradient descent, that is why we need a learning rate
    #In this case Adam will use gradient boosting with BinaryCrossEntropy and it's tryes to minimice binary_accuracy
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    #loss function used with Adam optimizer: BinaryCrossentropy
    #with from_logits=True we get a more stable result
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    #metric that optimizer will measure and will try to minimize using the loss function: binary_accuracy
    metrics = keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None)
    #model compile, it's like intitialice the model before training it 
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


#Set initial standard hyperparameters and made an inital model as a reference
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/v1/horse_vs_humans_v1_{epoch:02d}_{binary_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max')

input_size=150
learning_rate=0.01
size_inner = 100
dropout = 0
model = make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size_inner, droprate=dropout)

#####-----------LEARNING RATE---------------####
###Choosing best Learning Rate with Adam Optimizer using BinaryCrossentropy function and BinaryAcc as metric

##CHECKPOINTS
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/v2/horse_vs_humans_v2_{epoch:02d}_{binary_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max')

##MODEL INPUTS
input_size=150
size_inner = 10
dropout = 0
learning_rates = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0001]

##TESTING LEARNING RATES
scores = {}

for lr in learning_rates:
    model = make_model(input_size=input_size, learning_rate=lr, size_inner=size_inner, droprate=dropout)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])
    scores[lr] = history.history
# #PLOT RESULTS 
# for lr, hist in scores.items():
#     print("------------------------")
#     plt.figure()
#     plt.title(f'Validation Accuracy and Training Accuracy vs Epoch for Learning Rate {lr}')
#     plt.plot(hist["val_binary_accuracy"], label='Validation acc')
#     plt.plot(hist["binary_accuracy"], label='Training acc')
#     plt.ylabel("accuracy")
#     plt.xlabel("epoch")
#     plt.legend()
#     print("------------------------")


#####-----------INER SIZE---------------####
###Choosing Inner Size of Dense Layer

##CHECKPOINTS
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/v3/horse_vs_humans_v3_{epoch:02d}_{binary_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max')

##MODEL INPUTS
input_size=150
learning_rate = 0.025 #best in prev step consider graph study
dropout = 0 
size_inner_array = [10, 50, 100, 150]

##TESTING LEARNING RATES
scores = {}
for size_inner in size_inner_array:
    model = make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size_inner, droprate=dropout)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])
    scores[size_inner] = history.history
#PLOT RESULTS 
# for size_inner, hist in scores.items():
#     print("------------------------")
#     plt.figure()
#     plt.title(f'Validation Accuracy and Training Accuracy vs Epoch for Size Inner {size_inner}')
#     plt.plot(hist["val_binary_accuracy"], label='Validation acc')
#     plt.plot(hist["binary_accuracy"], label='Training acc')
#     plt.ylabel("accuracy")
#     plt.xlabel("epoch")
#     plt.legend()
#     print("------------------------")


#####-----------DROPOUT---------------####
###Choosing Inner Size of Dense Layer

##CHECKPOINTS
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/v4/horse_vs_humans_v4_{epoch:02d}_{binary_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max')

##MODEL INPUTS
input_size=150
learning_rate = 0.025 #best in prev step consider graph study
dropout_array = [0, 0.1, 0.2, 0.4, 0.6, 0.8]
size_inner_array = 150

##TESTING LEARNING RATES
scores = {}
for dropout in dropout_array:
    model = make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size_inner, droprate=dropout)
    history = model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[checkpoint])
    scores[dropout] = history.history
#PLOT RESULTS 
# for dropout, hist in scores.items():
#     print("------------------------")
#     plt.figure()
#     plt.title(f'Validation Accuracy and Training Accuracy vs Epoch for Dropout {dropout}')
#     plt.plot(hist["val_binary_accuracy"], label='Validation acc')
#     plt.plot(hist["binary_accuracy"], label='Training acc')
#     plt.ylabel("accuracy")
#     plt.xlabel("epoch")
#     plt.legend()
#     print("------------------------")


#####-----------FINAL MODEL---------------####

##CHECKPOINTS
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/v5/horse_vs_humans_v5_{epoch:02d}_{binary_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max')


##MODEL INPUTS
input_size=150
learning_rate = 0.025 #best in prev steps consider graph study
dropout = 0.2 #best in prev steps consider graph study
size_inner = 150 #best in prev steps consider graph study

##GET FINAL MODEL##
final_model = make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size_inner, droprate=dropout)
history = final_model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[checkpoint])

# plt.figure()
# plt.title(f'Validation Accuracy and Training Accuracy vs Epoch for Learining Rate={learning_rate}. Size Inner={size_inner}, Dropout={dropout}')
# plt.plot(history.history["val_binary_accuracy"], label='Validation acc')
# plt.plot(history.history["binary_accuracy"], label='Training acc')
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend()