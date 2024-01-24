
import os
import keras
from keras import layers
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

img_dir = "/kaggle/input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/"
for expression in os.listdir(img_dir):
    print(expression, "folder contians\t\t", len(os.listdir(img_dir + expression)), 'images')
BATCH_SIZE = 64
IMAGE_SIZE = 150
input_shape = (150, 150, 1)
Data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
train_gen = Data_gen.flow_from_directory(img_dir,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=BATCH_SIZE,
                                         color_mode="grayscale",
                                         shuffle=True,
                                         class_mode="binary",
                                         subset="training")
valid_gen = Data_gen.flow_from_directory(img_dir,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=BATCH_SIZE,
                                         color_mode="grayscale",
                                         shuffle=False,
                                         class_mode="binary",
                                         subset="validation")
labels = train_gen.class_indices
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")])
model.compile(

    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy'])
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("Model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(train_gen, verbose=1, callbacks=[early_stopping, checkpoint], epochs=20, validation_data=valid_gen)
train_loss, train_acc = model.evaluate(train_gen)
test_loss, test_acc = model.evaluate(valid_gen)
# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()