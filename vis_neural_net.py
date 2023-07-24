import keras.losses
import tensorflow as tf
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import CSVLogger
import numpy as np
import os
import pathlib
import visualkeras
from PIL import ImageFont
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(28, 28), name='Flatten'))
    model.add(layers.Dense(256, activation='relu', name='Dense'))
    model.add(layers.Dense(10, activation='softmax', name='Output'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger('plots_logs/training.log', separator=',', append=False)
    model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=2, validation_data=(x_test, y_test), callbacks=[csv_logger])

    model.save('model.keras')


def plot():
    log_data = pd.read_csv('plots_logs/training.log', sep=',', engine='python')
    accuracy = log_data['accuracy']
    val_accuracy = log_data['val_accuracy']
    epochs = range(1, 31)
    plt.clf()
    plt.plot(epochs, accuracy)
    plt.plot(epochs, val_accuracy)
    plt.title("Training and test accuracy evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Training', 'Evaluation'], loc='lower right')
    plt.savefig('plots_logs/accuracy_plot.png')


def val(img_path, model, number):
    image = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    input_arr = img_to_array(image).astype('float32') / 255.0

    plt.imshow(input_arr, cmap='Pastel1_r')
    plt.savefig(f"plots_logs/{number}.png")

    input_arr = np.reshape(input_arr, (1, 28, 28))

    predictions = model.predict(input_arr)
    predicted_class = np.argmax(predictions)

    print(f"Predicted class: {predicted_class}")


def visualize(model):
    print(model.summary())

    font = ImageFont.truetype("arial.ttf", 16)
    color_map = defaultdict(dict)
    color_map[layers.Flatten]['fill'] = 'lightcoral'
    color_map[layers.Dense]['fill'] = 'powderblue'

    visualkeras.layered_view(model, legend=True, font=font, scale_xy=1, one_dim_orientation='y', color_map=color_map, to_file="plots_logs/model.png")


def main():

    loaded_model = tf.keras.models.load_model('model.keras')
    for file in os.listdir(os.path.join(pathlib.Path(__file__).parent, 'test_images')):
        val(os.path.join(pathlib.Path(__file__).parent, 'test_images', file), loaded_model, file[0])
    visualize(loaded_model)
    plot()


if __name__ == '__main__':
    main()
