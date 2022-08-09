import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
import shutil
import random
import scipy
import numpy as np

# LOAD AND PROCESS IMAGES WITH IMAGEDATAGENERATOR
DATASET_PATH = './dataset'
TRAINING_PATH = './dataset/training'
TESTING_PATH = './dataset/testing'

train_datagen = ImageDataGenerator(rescale=1/255)
train_data = train_datagen.flow_from_directory(TRAINING_PATH,
                                               target_size=(224,224),
                                               batch_size=8,
                                               shuffle=True,
                                               class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1/255)
test_data = test_datagen.flow_from_directory(TESTING_PATH,
                                             target_size=(224,224),
                                             batch_size=8,
                                             shuffle=True,
                                             class_mode='binary')

# VISUALIZING LOADED IMAGES
print(train_data.class_indices)
imgs, labels = next(train_data)
fig, ax = plt.subplots(2,4)
ax = ax.ravel()
name_classes = list(train_data.class_indices.keys())
for i in range(8):
    ax[i].imshow(imgs[i,:,:,:])
    label = int(labels[i])
    title_name = name_classes[label]
    ax[i].set_title(title_name)
    ax[i].axis('off')

# CREATING THE CNN MODEL
model = Sequential([
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=8, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# DEFINE THE OPTIMIZER AND LOSS
model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# FIT THE MODEL
history = model.fit(
        train_data,
        epochs=30,
        validation_data=test_data
)

# VISUALIZING LOSS AND ACCURACY RESULT
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# VERIFY WITH OTHER IMAGES
path_file = './images.jpg'
def do_prediction(path_file):
    img = tf.io.read_file(path_file)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size=[224,224])
    img = img / 255.
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_num_class = int(np.round(prediction))
    predicted_class = name_classes[predicted_num_class]
    plt.imshow(img[0,:,:,:])
    plt.axis('off')
    if predicted_num_class == 0:
        string_prediction = str(1 - prediction[0])
    else:
        string_prediction = str(prediction[0])
    plt.title(predicted_class + ' with confidence= ' + str(string_prediction))

do_prediction(path_file)