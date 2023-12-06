# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:15:58 2021

@author: igonzalez2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory


#%% Constants

shapes_list = ['cube','cylinder','sphere','silo']
IMG_SIZE = (64,64)
BATCH_SIZE = 32


#%% Use the trained model

import_path = "./models/3D_V150000"

model_reloaded = tf.keras.models.load_model(import_path)


#%% Our data 

real_dir = './datasets/primitives/3D/Internet'

real_dataset = image_dataset_from_directory(real_dir,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           image_size=IMG_SIZE)


#%% Make predictions

probability_model = tf.keras.Sequential([model_reloaded, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(real_dataset)


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
  plt.yticks()
  thisplot = plt.bar(shapes_list, predictions_array, color="#777777")
  plt.xticks(rotation=45)
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#%% Plot single image

i = 5
for images, labels in real_dataset.take(1):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], labels, images.numpy().astype("uint8"))
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  labels)
    plt.show()


#%% Plot multiple

num_rows = 3
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    for images, labels in real_dataset.take(1):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], labels, images.numpy().astype("uint8"))
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i],  labels)
plt.tight_layout()
plt.show()


#%% Evaluate accuracy

val_loss, val_acc = model_reloaded.evaluate(real_dataset)

print('\nTest accuracy:', val_acc)