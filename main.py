import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Reshape, Conv2DTranspose, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist

# Caricamento del dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# normalizzazione e reshape
X_train = X_train / 255
X_train = X_train.reshape(len(X_train),28,28,1)
X_test = X_test / 255
X_test = X_test.reshape(len(X_test),28,28,1)

# Plot image
# w = 5
# h = 5
# fig, ax = plt.subplots(w, h)
# ax = ax.ravel()
# imgs, labels = next(train_data)
# for i in range(len(ax)):
#     idx = random.choice(np.arange(len(imgs)))
#     ax[i].imshow(imgs[idx,:,:,0], cmap='gray')
#     ax[i].axis('off')
#     ax[i].set_title(labels[idx])

# Definizione del modello CNN
model = Sequential([
                    # Encoder
                    Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
                    MaxPool2D(pool_size=(2,2)),
                    Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'),
                    MaxPool2D(pool_size=(2,2), padding='same'),
                    Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),

                    Flatten(),
                    # Decoder
                    Reshape((4,4,8)),

                    Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(filters=8, kernel_size=(3,3), activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
                    UpSampling2D((2,2)),
                    Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')
                    ])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, X_train, epochs=50)

autoencoder = model
encoder = tf.keras.models.Model(inputs = autoencoder.input,
                                outputs = autoencoder.get_layer('flatten').output)

decoder = tf.keras.models.Model(inputs = autoencoder.get_layer('reshape').input,
                                outputs = autoencoder.output)

coded_test_imgs = encoder.predict(X_test)
decoded_test_imgs = decoder(coded_test_imgs)

# Visualizzare 10 immagini a caso
n_imgs = 10
index_imgs =np.random.randint(0,X_test.shape[0],n_imgs)
fig, ax = plt.subplots(3,n_imgs, figsize=(18,18))
#ax = ax.ravel()
for i, index_img in enumerate(index_imgs):
    # Immagini originali
    ax[0,i].imshow(X_test[index_img,:,:,0], cmap='gray')
    ax[0,i].axis('off')

    # Immagini codificate
    ax[1, i].imshow(coded_test_imgs[index_img].reshape((16,8)), cmap='gray')
    ax[1, i].axis('off')

    # Immagini codificate
    ax[2, i].imshow(decoded_test_imgs[index_img,:,:,0], cmap='gray')
    ax[2, i].axis('off')