import keras.losses
import tensorflow as tf
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import pathlib


def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=2, validation_data=(x_test, y_test))

    model.save('model.keras')


def val(img_path):
    loaded_model = tf.keras.models.load_model('model.keras')

    image = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    input_arr = img_to_array(image).astype('float32') / 255.0
    input_arr = np.reshape(input_arr, (1, 28, 28))
    predictions = loaded_model.predict(input_arr)
    predicted_class = np.argmax(predictions)

    print(f"Predicted class: {predicted_class}")


def main():
    for file in os.listdir(os.path.join(pathlib.Path(__file__).parent, 'test_images')):
        val(os.path.join(pathlib.Path(__file__).parent, 'test_images', file))


if __name__ == '__main__':
    main()
