import tensorflow as tf
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=8, batch_size=64, validation_data=(x_test, y_test))

    model.save('model.keras')


def val(img_path):
    loaded_model = tf.keras.models.load_model('model.keras')

    img_path = img_path
    image = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    input_arr = 1.0 - (np.expand_dims(img_to_array(image), axis=0)).astype('float32') / 255.0
    predictions = loaded_model.predict(input_arr)
    predicted_class = np.argmax(predictions)

    print(f"Predicted class: {predicted_class}")


def main():
    train_model()
    val("C:/Users/Zenov/Documents/3.png")
    val("C:/Users/Zenov/Documents/1.png")
    val("C:/Users/Zenov/Documents/5.png")
    val("C:/Users/Zenov/Documents/7.png")
    val("C:/Users/Zenov/Documents/8.png")
    val("C:/Users/Zenov/Documents/9.png")


if __name__ == '__main__':
    main()
