import math
from re import S
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, Concatenate, Activation
from tensorflow.keras.activations import sigmoid, relu

class CNN_2Stream(tf.keras.Model):
    def __init__(self, class_num, include_top=True):
        super(CNN_2Stream, self).__init__()
        weight_init = TruncatedNormal(stddev=0.01)      # TruncatedNormal -> 랜덤하게 값을 불러오지만 너무 작거나 큰 값은 가져오지 않음 -> 모델에 너무 작거나 큰 값이 들어올 경우 문제가 발생할 수 있음
        bias_init = Constant(0.01)
        self.include_top = include_top

        self.conv1 = Conv1D(512, kernel_size=7, padding='same', kernel_initializer=weight_init, bias_initializer=bias_init)
        self.maxpool1 = MaxPool1D(pool_size=3, strides=3)
        self.sigmoid = Activation('sigmoid')

        self.conv2 = Conv1D(512, kernel_size=7, padding='same')
        self.maxpool2 = MaxPool1D(pool_size=3, strides=3)
        self.relu1 = Activation('relu')

        self.conv3 = Conv1D(512, kernel_size=7, padding='same')
        self.maxpool3 = MaxPool1D(pool_size=3, strides=3)
        self.relu2 = Activation('relu')

        self.concat = Concatenate(axis=2)

        if include_top:
            self.fc = Sequential([
                Flatten(),
                Dense(512, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init),
                Dropout(0.05),
                Dense(class_num, activation='softmax')
            ])

    def call(self, x):
        x1 = x[:, :, :3]
        x1 = self.conv1(x1)
        x1 = self.maxpool1(x1)
        x1 = self.sigmoid(x1)
        x1 = self.conv2(x1)
        x1 = self.maxpool2(x1)
        x1 = self.relu1(x1)
        x1 = self.conv3(x1)
        x1 = self.maxpool3(x1)
        x1 = self.relu2(x1)

        x2 = x[:, :, 3:]
        x2 = self.conv1(x2)
        x2 = self.maxpool1(x2)
        x2 = self.sigmoid(x2)
        x2 = self.conv2(x2)
        x2 = self.maxpool2(x2)
        x2 = self.relu1(x2)
        x2 = self.conv3(x2)
        x2 = self.maxpool3(x2)
        x2 = self.relu2(x2)

        x = self.concat([x1, x2])
        
        if self.include_top:
            logits = self.fc(x)
        else:
            logits = x

        return logits

