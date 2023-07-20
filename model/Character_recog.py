import numpy as np
import csv
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#Đọc dữ liệu
with open('hand_written.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []
    
    # Đọc từng dòng của file và thêm vào list rows, mỗi phần tử của list là một dòng
    for row in result:
        rows.append(row)

train_data = [] # Dữ liệu training
train_label = [] # Label của chúng

# Lặp qua dữ liệu đã lưu trong danh sách rows
for letter in rows:
    # if (letter[0] == '0') or (letter[0] == '1') or (letter[0] == '2') or (letter[0] == '3'):
        x = np.array([int(j) for j in letter[1:]])
        x = x.reshape(28, 28)
        train_data.append(x)
        train_label.append(int(letter[0]))
    # else:
        # break

print(len(train_label)) #result: 372451
import random

shuffle_order = list(range(372451))
random.shuffle(shuffle_order)

train_data = np.array(train_data)
train_label = np.array(train_label)

train_data = train_data[shuffle_order]
train_label = train_label[shuffle_order]

print(train_data.shape)
train_x = train_data[:260715]
train_y = train_label[:260715]

val_x = train_data[260715:316582]
val_y = train_label[260715:316582]

test_x = train_data[316582:]
test_y = train_label[316582:]

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical


BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 26
LR = 0.001
N_EPOCHS = 50


network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) #1

network = conv_2d(network, 32, 3, activation='relu') #2
network = max_pool_2d(network, 2) #3

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu') #4
network = dropout(network, 0.8) #5

network = fully_connected(network, N_CLASSES, activation='softmax')#6
network = regression(network)

model = tflearn.DNN(network) #7

train_x = train_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_x = val_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

original_test_y = test_y #được sử dụng để test ở bước sau

train_y = to_categorical(train_y, N_CLASSES)
val_y = to_categorical(val_y, N_CLASSES)
test_y = to_categorical(test_y, N_CLASSES)
# with tf.compat.v1.Session() as sess:
#     # Khôi phục model từ checkpoint
#     # model.load('handwring.tflearn')
model.fit(train_x, train_y, n_epoch=N_EPOCHS, validation_set=(val_x, val_y), show_metric=True)

model.save('handwring.tflearn')
model.load('handwring.tflearn')

# dự đoán với tập dữ liệu test
test_logits = model.predict(test_x)
#lấy phần tử có giá trị lớn nhất 
test_logits = np.argmax(test_logits, axis=-1)
print(np.sum(test_logits == original_test_y) / len(test_logits))
