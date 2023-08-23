from keras.utils import to_categorical
import pickle
from random import sample
import cv2
from numpy import empty
from numpy import uint8
import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import concatenate
from keras.layers import Conv2D , MaxPool2D , Input ,AveragePooling2D, Dense , Dropout ,Activation, Flatten , BatchNormalization , ZeroPadding2D , MaxPooling2D
from matplotlib import pyplot as plt

def IdentityBlock(prev_Layer , filters):
    f1 , f2 , f3 = filters

    x = Conv2D(filters=f1, kernel_size = (1,1) , strides=(1,1), padding='valid')(prev_Layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    
    x = Conv2D(filters=f2, kernel_size = (3,3) , strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    
    x = Conv2D(filters=f3, kernel_size = (1,1) , strides=(1,1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    
    x = concatenate([ x, prev_Layer ], axis=-1)
    x = Activation(activation='relu')(x)
    return x   

def ConvBlock(prev_Layer , filters , strides):
    f1 , f2 , f3 = filters
    
    #Path 1
    x = Conv2D(filters=f1, kernel_size = (1,1) ,padding='valid', strides=strides)(prev_Layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    
    x = Conv2D(filters=f2, kernel_size = (3,3) , padding='same' , strides=(1 ,1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    
    x = Conv2D(filters=f3, kernel_size = (1,1), padding='valid' , strides=(1 ,1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    
    #Path 2
    
    x2 = Conv2D(filters=f3, kernel_size=(1,1), padding='valid' , strides=strides)(prev_Layer)
    x2 = BatchNormalization(axis=3)(x2)
    
    x = concatenate([x , x2], axis=-1)
    x = Activation(activation='relu')(x)
    return x
 
def ResNet50():
    input_layer = Input(shape = (224, 224, 3))
    #Stage 1
    x = ZeroPadding2D((3, 3))(input_layer)
    x = Conv2D(filters = 64, kernel_size = (7,7), strides=(2,2)) (x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=(3,3) , strides=(2,2))(x)
    
    #Stage 2
    x = ConvBlock(prev_Layer=x, filters = [32 , 32 , 128], strides = 1)
    x = IdentityBlock(prev_Layer=x, filters = [32 , 32 , 128])
    x = IdentityBlock(prev_Layer=x, filters = [32 , 32 , 128])
    
    #Stage 3
    x = ConvBlock(prev_Layer=x, filters = [32 , 32 , 128], strides = 2)
    x = IdentityBlock(prev_Layer=x, filters = [32 , 32 , 128])
    x = IdentityBlock(prev_Layer=x, filters = [32 , 32 , 128])
    x = IdentityBlock(prev_Layer=x, filters = [32 , 32 , 128])

    #Stage 4    
    x = ConvBlock(prev_Layer=x, filters = [64 , 64 , 256], strides = 2)    
    x = IdentityBlock(prev_Layer=x, filters = [64 , 64 , 256])
    x = IdentityBlock(prev_Layer=x, filters = [64 , 64 , 256])
    x = IdentityBlock(prev_Layer=x, filters = [64 , 64 , 256])
    x = IdentityBlock(prev_Layer=x, filters = [64 , 64 , 256])
    x = IdentityBlock(prev_Layer=x, filters = [64 , 64 , 256])
    
    #Stage 5
    x = ConvBlock(prev_Layer=x, filters = [128 , 128 , 512], strides = 2)
    x = IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
    x = IdentityBlock(prev_Layer=x, filters = [128 , 128 , 512])
    
    #Stage 6
    x = AveragePooling2D(pool_size=(7,7)) (x)
    
    x = Flatten()(x)
    x = Dense(units=504, activation='softmax')(x)
    
    model = Model(inputs=input_layer , outputs = x , name='ResNet50')
    return model


train_no = 2012 * 5
test_no = 503 * 5

height = 224
width = 224
dim = (width, height)

training_matrix = empty([train_no, width, height, 3], dtype=uint8)
test_matrix = empty([test_no, width, height, 3], dtype=uint8)

cnt_test = 0
cnt_train = 0


for i in range(1, test_no + 1, 1):
    img = cv2.imread(
        "test_directory"+str(i)+".bmp")
    res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    test_matrix[cnt_test] = res

    cnt_test += 1

for i in range(1, train_no + 1, 1):
    img = cv2.imread(
        "train_directory"+str(i)+".bmp")
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

model = ResNet50()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = model.fit(X_train, y_train, validation_split = 0.2, epochs=50, batch_size=30)
loss, acc, pre, rec = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
print("Overall Accuracy: ", acc * 100)
print("Overall Precision: ", pre * 100)
print("Overall Recall: ", rec * 100)

file = open("y_pred_RES_major", "wb")
pickle.dump(y_pred, file)
file.close()
file = open("y_test_RES_major", "wb")
pickle.dump(y_test, file)
file.close()

file = open("trainHistoryRES_major", "wb")
pickle.dump(history, file)
file.close()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


