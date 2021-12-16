import math
from re import S
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Dense, Dropout, Flatten, Activation

class RT_CNN(tf.keras.Model):
    def __init__(self, class_num, include_top=True):
        super(RT_CNN, self).__init__()
        weight_init = TruncatedNormal(stddev=0.01)      # TruncatedNormal -> 랜덤하게 값을 불러오지만 너무 작거나 큰 값은 가져오지 않음 -> 모델에 너무 작거나 큰 값이 들어올 경우 문제가 발생할 수 있음
        bias_init = Constant(0.01)
        self.include_top = include_top

        self.conv1 = Conv1D(512, kernel_size=7, padding='same', kernel_initializer=weight_init, bias_initializer=bias_init)
        # self.bn1 = BatchNormalization()
        self.maxpool1 = MaxPool1D(pool_size=3, strides=3)
        self.relu1 = Activation('relu')

        self.conv2 = Conv1D(512, kernel_size=7, padding='same')
        # self.bn2 = BatchNormalization()
        self.maxpool2 = MaxPool1D(pool_size=3, strides=3)
        self.relu2 = Activation('relu')

        self.conv3 = Conv1D(512, kernel_size=7, padding='same')
        # self.bn3 = BatchNormalization()
        self.maxpool3 = MaxPool1D(pool_size=3, strides=3)
        self.relu3 = Activation('relu')

        if include_top:
            self.fc = Sequential([
                Flatten(),
                Dense(512, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init),
                Dropout(0.05),
                # Dense(256, activation='relu'),
                # Dropout(0.05),
                # Dense(class_num, activation='softmax')
                Dense(class_num, activation='softmax')
            ])

    def call(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)
        
        if self.include_top:
            logits = self.fc(x)
        else:
            logits = x

        return logits

