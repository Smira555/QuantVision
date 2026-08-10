# Importing all the necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import pathlib
import PIL
import PIL.Image
from tensorflow import keras

train_data_dir = pathlib.Path('Train')
test_data_dir = pathlib.Path('Test')

train_image_count = len(list(train_data_dir.glob('*/*.png')))   
print(train_image_count)
test_image_count = len(list(test_data_dir.glob('*/*.png')))
print(test_image_count)

train_up = list(train_data_dir.glob('Up/*'))
test_up = list(test_data_dir.glob('Up/*'))
train_down = list(train_data_dir.glob('Down/*'))
test_down = list(test_data_dir.glob('Down/*'))

img = PIL.Image.open(str(train_up[0]))
plt.imshow(img)
plt.axis('off')
# plt.show()

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    batch_size = batch_size,
    seed = 42,
    image_size = (img_height,img_width)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_data_dir,
    batch_size = batch_size,
    seed = 42,
    image_size = (img_height,img_width)
)

class_names = train_ds.class_names
print(class_names)

autotune = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=autotune)
test_ds = test_ds.cache().prefetch(buffer_size=autotune)

for image_batch,label_batch in train_ds:
    print(image_batch.shape)
    print(label_batch.shape)
    break

# Initially, without any augmentation, the data model would simply overfit.
# And adding all the augmentations would cause the model to just predict up or down causing the accuracy to be about 55 percent.
# I checked all the augmentations one by one to check which one was causing the model to go hey via and commented them out.
# I checked if model overfitted or just gave 0.5 accuracy for all epochs. If Overfitted, I added more augmentation if not then I minimized it.
# This is the final augmentation that worked best.
augmentation = tf.keras.Sequential(
    [tf.keras.layers.RandomZoom(0.02),
 
#  tf.keras.layers.RandomRotation(0.05),

#  tf.keras.layers.RandomContrast(0.05),
 
 tf.keras.layers.RandomTranslation(height_factor = 0.02, width_factor = 0.02),

#  tf.keras.layers.RandomBrightness(0.02),

 ]
)

fully_connect_num = 16
# Model:

# I made the dropout value to be 0.5 to prevent easy overfitting of model by making it hard to memorize
# I also made the model very small so that it cannot easily overfit.
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    augmentation,
    tf.keras.layers.Conv2D(16,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.GaussianNoise(0.15),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(fully_connect_num, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
met = metrics = ['accuracy']
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = met
)

training = model.fit(train_ds, validation_data = test_ds, epochs = 50)

# acc = training.history['accuracy']
# print(acc)

# Overall, The dataset is too small for CNNs. a little harsh data augmentation leads to Overfitting.
# And too less of data augmentation causes Model to spit out just one value
# I have even used 0.5 dropout value to prevent overfitting, and reduced the model size significantly
# This is the closest I have gotten so that the model does not overfit and does not spit out just one value.