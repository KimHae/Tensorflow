import tensorflow as tf
from tensorflow import keras
import os
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("not trained model's accuracy: {:5.2f}%".format(100*acc))

#가중치를 복원하기
checkpoint = "training_1/cp.ckpt"
model.load_weights(checkpoint)
loss, acc = model.evaluate(test_images, test_labels)
print("restored model's accuracy: {:5.2f}%".format(100*acc))