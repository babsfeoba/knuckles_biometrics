# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:57:10 2022

@author: Cmpe238
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle
from random import sample
from numpy import  empty, uint8
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
import cv2

train_no = 2012 * 5
test_no = 503 * 5

height = 227
width = 227
dim = (width, height)

training_matrix = empty([train_no, width, height, 3], dtype=uint8)
test_matrix = empty([test_no, width, height, 3], dtype=uint8)

cnt_test = 0
cnt_train = 0


for i in range(1, test_no + 1, 1):
    img = cv2.imread(
        "C:/Users/usr310/Documents/Knuckles/Minor/Gen_Test/"+str(i)+".bmp")
    res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    test_matrix[cnt_test] = res

    cnt_test += 1

for i in range(1, train_no + 1, 1):
    img = cv2.imread(
        "C:/Users/usr310/Documents/Knuckles/Minor/Gen_Train/"+str(i)+".bmp")
    res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    training_matrix[cnt_train] = res

    cnt_train += 1

train_label = [(i//20)+1 if not i % 20 == 0 else (i//20)
               for i in range(1, cnt_train + 1, 1)]
test_label = [(i//5)+1 if not i % 5 == 0 else (i//5)
              for i in range(1, cnt_test + 1, 1)]
random_numbers = sample(range(cnt_train), cnt_train)
random_numbers2 = sample(range(cnt_test), cnt_test)

file = open("random_numbersO", "wb")
pickle.dump(random_numbers, file)
file.close()
training_matrix_shuffled = empty([train_no, width, height, 3], dtype=uint8)

file = open("random_numbersO2", "wb")
pickle.dump(random_numbers2, file)
file.close()
test_matrix_shuffled = empty([test_no, height, width, 3], dtype=uint8)
#
for i in range(len(train_label)):
    training_matrix_shuffled[i] = training_matrix[random_numbers[i]]

for i in range(len(test_label)):
    test_matrix_shuffled[i] = test_matrix[random_numbers2[i]]

#
train_label_shuffled = [None for i in range(train_no)]
test_label_shuffled = [None for i in range(test_no)]
#
for i in range(len(random_numbers)):
    train_label_shuffled[i] = train_label[random_numbers[i]]
for i in range(len(random_numbers2)):
    test_label_shuffled[i] = test_label[random_numbers2[i]]


X_train = training_matrix_shuffled.reshape(-1, height, width, 3)

X_test = test_matrix_shuffled.reshape(-1, height, width, 3)

y_train = to_categorical(train_label_shuffled)
y_test = to_categorical(test_label_shuffled)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
          input_shape=(height, width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))

# model.add(Dropout(0.3))
model.add(Dense(504, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= tf.keras.optimizers.Adam(),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = model.fit(X_train, y_train, validation_split = 0.25, epochs=10, batch_size=30)
loss, acc, pre, rec = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
print("Overall Accuracy: ", acc * 100)
print("Overall Precision: ", pre * 100)
print("Overall Recall: ", rec * 100)

file = open("y_pred_AN_minor", "wb")
pickle.dump(y_pred, file)
file.close()
file = open("y_test_AN_minor", "wb")
pickle.dump(y_test, file)
file.close()


file = open("trainHistoryAN_minor", "wb")
pickle.dump(history, file)
file.close()

from keras.models import load_model
model.save('model.h5')
model_final = load_model('model.h5')

from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
