import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence

def Read_Data(dataset, seg_size):
    data_path = os.path.join('Data', dataset)
    train_X = np.load(data_path+'train_x.npy')
    train_Y = np.load(data_path+'train_y.npy')
    test_X = np.load(data_path+'val_x.npy')
    test_Y = np.load(data_path+'val_y.npy')
    
    train_X = tf.convert_to_tensor(train_X, dtype=tf.float32)
    train_X = tf.reshape(train_X, [-1, seg_size, 6])
    train_Y = tf.convert_to_tensor(train_Y, dtype=tf.float32)
    test_X = tf.convert_to_tensor(test_X, dtype=tf.float32)
    test_X = tf.reshape(test_X, [-1, seg_size, 6])
    test_Y = tf.convert_to_tensor(test_Y, dtype=tf.float32)
    
    return train_X, train_Y, test_X, test_Y


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt