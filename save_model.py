from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

def create_model():
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
    return model
'''
#체크포인트를 리용한 모델저장,매학슴단계마다 콜백시켜 가중치를 계속 저장한다.
checkpoint_path = "training_1/cp.ckpt"#폴더가 자동으로 생기므로 폴도만들기 조작이 없다
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)
model = create_model()
model.fit(train_images, train_labels,  epochs = 5, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training
loss,acc = model.evaluate(test_images,test_labels)
#result = model.evaluate(test_images,test_labels)
model.load_weights('training_1/cp.ckpt')
loss,acc = model.evaluate(test_images, test_labels)
print("callback restored model's accuracy: {:5.2f}%".format(100*acc))


#수동으로 가중치 저장하기
model.save_weights('checkpoints/my_checkpoint')#폴더가 자동으로 생긴다.
model = create_model()
model.load_weights('checkpoints/my_checkpoint')
loss,acc = model.evaluate(test_images, test_labels)
print("handle restored model's accuracy: {:5.2f}%".format(100*acc))
'''
#모델전체를 저장하기
model = create_model()
model.fit(train_images,train_labels,epochs=5)
model.save('totalmodel/my_model.h5')#폴더를 자체로 만들지못하므로 폴더만들기 조작이 필요하다.
new_model = keras.models.load_model('totalmodel/my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("total saved model's accuracy: {:5.2f}%".format(100*acc))