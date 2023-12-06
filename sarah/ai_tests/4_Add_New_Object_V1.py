# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:13:38 2021

@author: igonzalez2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
# import time

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#%% Constants

shapes_list = ['cube','cylinder','sphere','silo']
IMG_SIZE = (64,64)
BATCH_SIZE = 32
n_samples = 500 #Up to 480000
index = np.arange(480000)


#%% Import Dataset

f = h5py.File('./datasets/primitives/3D/3dshapes.h5', 'r')

index = random.sample(list(index), n_samples) #To take random samples from the dataset

images = f['images'][sorted(index)]  # array shape [n_samples,64,64,3], uint8 in range(256)
labels = f['labels'][sorted(index),4]  # array shape [n_samples,], float64

images3shapes = []
labels3shapes = []
for a in range(n_samples):
    if labels[a] == 3 : pass # 0 for exclude cube, 1 for cylinder, 2 for sphere et 3 for silo
    else : 
        labels3shapes.append(labels[a])
        images3shapes.append(images[a])
        
labels3shapes = np.array(labels3shapes, dtype=np.float)   
images3shapes = np.array(images3shapes, dtype=np.uint8)    

f.close()


#%% Divide into 80% Train - 5% Test - 15% Validation
#All
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.20)
images_test, images_validation, labels_test, labels_validation = train_test_split(images_test, labels_test, test_size=0.75)

#3shapes
images_3shapes_train, images_3shapes_test, labels_3shapes_train, labels_3shapes_test = train_test_split(images3shapes, labels3shapes, test_size=0.20)
images_3shapes_test, images_3shapes_validation, labels_3shapes_test, labels_3shapes_validation = train_test_split(images_3shapes_test, labels_3shapes_test, test_size=0.75)

    
#%% Preprocess the data
#All
train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test)).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((images_validation, labels_validation)).batch(BATCH_SIZE)

#3Shapes
train_3shapes_dataset = tf.data.Dataset.from_tensor_slices((images_3shapes_train, labels_3shapes_train)).batch(BATCH_SIZE)
test_3shapes_dataset = tf.data.Dataset.from_tensor_slices((images_3shapes_test, labels_3shapes_test)).batch(BATCH_SIZE)
validation_3shapes_dataset = tf.data.Dataset.from_tensor_slices((images_3shapes_validation, labels_3shapes_validation)).batch(BATCH_SIZE)

#%% Configure the dataset for performance

AUTOTUNE = tf.data.experimental.AUTOTUNE

#All
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#3Shapes
train_3shapes_dataset = train_3shapes_dataset.prefetch(buffer_size=AUTOTUNE)
validation_3shapes_dataset = validation_3shapes_dataset.prefetch(buffer_size=AUTOTUNE)
test_3shapes_dataset = test_3shapes_dataset.prefetch(buffer_size=AUTOTUNE)


#%% Data augmentation

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(IMG_SIZE[0], 
                                                              IMG_SIZE[1],
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
  ]
)


plt.figure(figsize=(10, 10))
for im, _ in train_dataset.take(1):
  for i in range(9):
    augmented_images = data_augmentation(im)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


#%% Show images

indices =  random.sample(range(0, n_samples), 25) #25 random numbers between 0 and n_samples
plt.figure(figsize=(10,10))
a = 0
for i in indices:
    plt.subplot(5,5,a+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(shapes_list[int(labels[i])])
    a = a+1
plt.show()


#%%   Create the base model 3shapes

num_classes = 3

model3shapes = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


#%% Compile the base model

model3shapes.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model3shapes.summary()


#%% Training the base neural network

epochs = 15

history = model3shapes.fit(train_3shapes_dataset, validation_data=validation_3shapes_dataset, epochs=epochs)


#%% Evaluate accuracy

test_loss, test_acc = model3shapes.evaluate(test_3shapes_dataset)

print('\nTest accuracy:', test_acc)


#%% Visualize training results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#%% Make predictions

probability_model = tf.keras.Sequential([model3shapes, 
                                         tf.keras.layers.Softmax()])

predictions_3shapes = probability_model.predict(images_3shapes_test)


#%% Information from samples

cube_count_test = 0
cylinder_count_test = 0
sphere_count_test = 0
silo_count_test = 0

for i in labels_3shapes_test:
    if i == 0. : cube_count_test += 1
    if i == 1. : cylinder_count_test += 1
    if i == 2. : sphere_count_test += 1
    if i == 3. : silo_count_test += 1
    
cube_count_train = 0
cylinder_count_train = 0
sphere_count_train = 0
silo_count_train = 0

for i in labels_3shapes_train:
    if i == 0. : cube_count_train += 1
    if i == 1. : cylinder_count_train += 1
    if i == 2. : sphere_count_train += 1
    if i == 3. : silo_count_train += 1
    
print("Cube: {:.1f}%".format(cube_count_train/(0.8*n_samples)*100))
print("Cylinder: {:.1f}%".format(cylinder_count_train/(0.8*n_samples)*100))
print("Sphere: {:.1f}%".format(sphere_count_train/(0.8*n_samples)*100))
print("Silo: {:.1f}%".format(silo_count_train/(0.8*n_samples)*100))


#%% Statistics 

cc = 0  #cube when cube
cyc = 0 #cylinder when cube
spc = 0 #sphere when cube
sc = 0 #silo when cube

ccy = 0  #cube when cylinder
cycy = 0 #cylindre when cylinder
spcy = 0 #sphere when cylinder
scy = 0 #silo when cylinder

csp = 0  #cube when sphere
cysp = 0 #cylinder when sphere
spsp = 0 #sphere when sphere
ssp = 0 #silo when sphere

cs = 0  #cube when silo
cys = 0 #cylinder when silo
sps = 0 #sphere when silo
ss = 0 #silo when silo


for i in range(labels_3shapes_test.size):
    predicted_label = np.argmax(predictions_3shapes[i])
    real_label = int(labels_3shapes_test[i])
    # real_label = i
    if real_label == 0 : 
        if predicted_label == 0 : cc += 1
        if predicted_label == 1 : cyc += 1
        if predicted_label == 2 : spc += 1
        if predicted_label == 3 : sc += 1
    if real_label == 1 : 
        if predicted_label == 0 : ccy += 1
        if predicted_label == 1 : cycy += 1
        if predicted_label == 2 : spcy += 1
        if predicted_label == 3 : scy += 1
    if real_label == 2 : 
        if predicted_label == 0 : csp += 1
        if predicted_label == 1 : cysp += 1
        if predicted_label == 2 : spsp += 1
        if predicted_label == 3 : ssp += 1
    if real_label == 3 : 
        if predicted_label == 0 : cs += 1
        if predicted_label == 1 : cys += 1
        if predicted_label == 2 : sps += 1
        if predicted_label == 3 : ss += 1

performance_3shapes = [[cc*100/cube_count_test,cyc*100/cube_count_test,spc*100/cube_count_test,sc*100/cube_count_test],
                       [ccy*100/cylinder_count_test,cycy*100/cylinder_count_test,spcy*100/cylinder_count_test,scy*100/cylinder_count_test],
                       [csp*100/sphere_count_test,cysp*100/sphere_count_test,spsp*100/sphere_count_test,ssp*100/sphere_count_test],
                       [cs*100,cys*100,sps*100,ss*100]] 


#%% Take model3shapes and use it as base model (export + import)



export_path = "./models/3D_3shapes_V{}".format(int(n_samples)) 
model3shapes.save(export_path)

base_model = tf.keras.models.load_model(export_path)
base_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output) #We take the base model without the prediction layer

base_model.trainable = False
base_model.summary()


#%% Build a model by chaining

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(4)

inputs = tf.keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
outputs = prediction_layer(x) # we add our new prediction layer with 4 classes

model = tf.keras.Model(inputs, outputs)


#%%  Compile the model

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


#%% Evaluate the model

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


#%% Train the model
initial_epochs = 5
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


#%% Learning plot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#%% Make predictions

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(images_test)


#%% Start Fine Tuning 
    
base_model.trainable = True # Un-freeze the top layers of the model

print("Number of layers in the base model: ", len(base_model.layers)) #Number of layers in the base model:  12

fine_tune_at = 10 # Fine-tune from this layer onwards 
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()


#%% Continue training the model 

fine_tune_epochs = 1
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)


#%% Learning curves

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.6, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#%% Verify the performance

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


#%% Make predictions

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(images_test)


#%% Information from samples

cube_count_test = 0
cylinder_count_test = 0
sphere_count_test = 0
silo_count_test = 0

for i in labels_test:
    if i == 0. : cube_count_test += 1
    if i == 1. : cylinder_count_test += 1
    if i == 2. : sphere_count_test += 1
    if i == 3. : silo_count_test += 1
    
cube_count_train = 0
cylinder_count_train = 0
sphere_count_train = 0
silo_count_train = 0

for i in labels_train:
    if i == 0. : cube_count_train += 1
    if i == 1. : cylinder_count_train += 1
    if i == 2. : sphere_count_train += 1
    if i == 3. : silo_count_train += 1
    
print("Cube: {:.1f}%".format(cube_count_train/(0.8*n_samples)*100))
print("Cylinder: {:.1f}%".format(cylinder_count_train/(0.8*n_samples)*100))
print("Sphere: {:.1f}%".format(sphere_count_train/(0.8*n_samples)*100))
print("Silo: {:.1f}%".format(silo_count_train/(0.8*n_samples)*100))


#%% Statistics 

cc = 0  #cube when cube
cyc = 0 #cylinder when cube
spc = 0 #sphere when cube
sc = 0 #silo when cube

ccy = 0  #cube when cylinder
cycy = 0 #cylindre when cylinder
spcy = 0 #sphere when cylinder
scy = 0 #silo when cylinder

csp = 0  #cube when sphere
cysp = 0 #cylinder when sphere
spsp = 0 #sphere when sphere
ssp = 0 #silo when sphere

cs = 0  #cube when silo
cys = 0 #cylinder when silo
sps = 0 #sphere when silo
ss = 0 #silo when silo


for i in range(labels_test.size):
    predicted_label = np.argmax(predictions[i])
    real_label = int(labels_test[i])
    # real_label = i
    if real_label == 0 : 
        if predicted_label == 0 : cc += 1
        if predicted_label == 1 : cyc += 1
        if predicted_label == 2 : spc += 1
        if predicted_label == 3 : sc += 1
    if real_label == 1 : 
        if predicted_label == 0 : ccy += 1
        if predicted_label == 1 : cycy += 1
        if predicted_label == 2 : spcy += 1
        if predicted_label == 3 : scy += 1
    if real_label == 2 : 
        if predicted_label == 0 : csp += 1
        if predicted_label == 1 : cysp += 1
        if predicted_label == 2 : spsp += 1
        if predicted_label == 3 : ssp += 1
    if real_label == 3 : 
        if predicted_label == 0 : cs += 1
        if predicted_label == 1 : cys += 1
        if predicted_label == 2 : sps += 1
        if predicted_label == 3 : ss += 1

performance = [[cc*100/cube_count_test,cyc*100/cube_count_test,spc*100/cube_count_test,sc*100/cube_count_test],
               [ccy*100/cylinder_count_test,cycy*100/cylinder_count_test,spcy*100/cylinder_count_test,scy*100/cylinder_count_test],
               [csp*100/sphere_count_test,cysp*100/sphere_count_test,spsp*100/sphere_count_test,ssp*100/sphere_count_test],
               [cs*100/silo_count_test,cys*100/silo_count_test,sps*100/silo_count_test,ss*100/silo_count_test]] 

