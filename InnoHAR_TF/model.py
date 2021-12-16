from re import S
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, GRU, Dropout, Flatten, Dense, Concatenate, Activation

class Inceptionlike_block(tf.keras.Model):
    def __init__(self):
        super(Inceptionlike_block, self).__init__()

        self.l1 = Conv1D(128, kernel_size=1)
        self.relu1 = Activation('relu')

        self.l21 = Conv1D(128, kernel_size=1)
        self.relu21 = Activation('relu')
        self.l22 = Conv1D(128, kernel_size=3, padding='same')
        self.relu22 = Activation('relu')

        self.l31 = Conv1D(128, kernel_size=1)
        self.relu31 = Activation('relu')
        self.l32 = Conv1D(128, kernel_size=5, padding='same')
        self.relu32 = Activation('relu')

        self.l41 = MaxPool1D(pool_size=3, strides=1, padding='same')
        self.l42 = Conv1D(128, kernel_size=1)
        self.relu42 = Activation('relu')

        self.concat = Concatenate(axis=2)

    def call(self, x):
        x1 = self.l1(x)
        x1 = self.relu1(x1)
        x21 = self.l21(x)
        x21 = self.relu21(x21)
        x22 = self.l22(x21)
        x22 = self.relu22(x22)
        x31 = self.l31(x)
        x31 = self.relu31(x31)
        x32 = self.l32(x31)
        x32 = self.relu32(x32)
        x41 = self.l41(x)
        x42 = self.l42(x41)
        x42 = self.relu42(x42)
        
        outputs = self.concat([x1, x22, x32, x42])

        return outputs


class InnoHAR(tf.keras.Model):
    def __init__(self, class_num, segment_size, include_top=True):
        super(InnoHAR, self).__init__()
        self.include_top = include_top

        self.conv1 = Inceptionlike_block()
        self.conv2 = Inceptionlike_block()
        self.conv3 = Inceptionlike_block()
        self.maxpool1 = MaxPool1D(pool_size=int(segment_size/2), strides=1)
        # self.maxpool1 = MaxPool1D(pool_size=3, strides=1)
        self.conv4 = Inceptionlike_block()
        self.maxpool2 = MaxPool1D(pool_size=int(segment_size/2), strides=1)
        # self.maxpool2 = MaxPool1D(pool_size=3, strides=1)
        self.dropout1 = Dropout(0.2)

        self.gru1 = GRU(128, return_sequences=True)
        self.dropout2 = Dropout(0.2)
        self.gru2 = GRU(128)
        self.dropout3 = Dropout(0.2)
        self.flatten = Flatten()
        
        if self.include_top:
            self.fc = Dense(class_num, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout1(x)

        x = self.gru1(x)
        x = self.dropout2(x)
        x = self.gru2(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        
        if self.include_top:
            logits = self.fc(x)
        else:
            logits = x

        return logits