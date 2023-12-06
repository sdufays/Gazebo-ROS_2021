# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:04:58 2021

@author: igonzalez2
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import time

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

f.close()


#%% Divide into 80% Train - 20% Test

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.20)


#%% Preprocess the data

train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test)).batch(BATCH_SIZE)


#%% Configure the dataset for performance

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


#%% Standardize the data
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


#%% Data augmentation

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(IMG_SIZE[0], 
                                                              IMG_SIZE[1],
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.5),
    layers.experimental.preprocessing.RandomZoom(0.4),
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


#%% Set up the layers

num_classes = 4

model = Sequential([
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

#%% Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


#%% Training the neural network

epochs=20

history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)


#%% Evaluate accuracy

test_loss, test_acc = model.evaluate(test_dataset)

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

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(images_test)


#%% Plot functions

# =============================================================================
# /Function: <plot_image: Plots an image its predicted label, and the true label. Blue if correct, red if not.>
# Inputs: <i, predictions_array, true_label, img>
# Outputs:<->/
# =============================================================================

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(shapes_list[int(predicted_label)],
                                100*np.max(predictions_array),
                                shapes_list[int(true_label)]),
                                color=color)

# =============================================================================
# /Function: <plot_value_array: Plots a bar chart with the values of the predictions for each class.>
# Inputs: <i, predictions_array, true_label>
# Outputs:<->/
# =============================================================================

def plot_value_array(i, predictions_array, true_label):
  true_label = int(true_label[i])
  plt.grid(False)
  plt.xticks(range(4))
  plt.yticks([])
  thisplot = plt.bar(range(4), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
  
#%% Verify predictions

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], labels_test, images_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  labels_test)
plt.show()


#%% Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], labels_test, images_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], labels_test)
plt.tight_layout()
plt.show()


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

performance = [[cc*100/cube_count_test,cyc*100/cube_count_test,spc*100/cube_count_test,sc*100/cube_count_test],[ccy*100/cylinder_count_test,cycy*100/cylinder_count_test,spcy*100/cylinder_count_test,scy*100/cylinder_count_test],[csp*100/sphere_count_test,cysp*100/sphere_count_test,spsp*100/sphere_count_test,ssp*100/sphere_count_test],[cs*100/silo_count_test,cys*100/silo_count_test,sps*100/silo_count_test,ss*100/silo_count_test]] 


#%% Export the model

t = time.time()

export_path = "./models/Iker/3D_SimpleV{}".format(int(n_samples)) # We save the model inside a folder called "3D_SimpleV150000" for the model trained with 150.000 images
# export_path = "C:/Users/igonzalez2/Documents/Python Scripts/models/Iker/3D_50000" # We can choose the name if we want
model.save(export_path)


#%% Reload and confirm export
reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(images_test)
reloaded_result_batch = reloaded.predict(images_test)

print(abs(reloaded_result_batch - result_batch).max())









