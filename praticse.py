import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labesl),(test_images,test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images,train_labesl,epochs=5)
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print("success")