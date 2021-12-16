import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow.python.ops.gen_math_ops import mod

import random
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import save_model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.compat import v1 as tfv1

from model import CNN_2Stream
from utils import Read_Data
from tf_lite_converter.adam import Adam
from tf_lite_converter.sgd import SGD
from tf_lite_converter.softmax_classifier_head import SoftmaxClassifierHead
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

device = 'A31'
user_type = 'Multi'

file_path = os.path.join('Target', device, user_type)
dataset = file_path + '/'
save_dir = '2_Stream_CNN/save/' + device + '/' + user_type
checkpoint_path = save_dir + "/weights/cp.ckpt"

epochs = 30
batch_size = 10

segment_size = 250
class_num = 4

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

train_x, train_y, test_x, test_y = Read_Data(dataset, segment_size)

f = open(save_dir + "/data_shape.txt", 'w')
f.write("Train data shape : " + str(train_x.shape) + "\n")
f.write("Validation data shape : " + str(test_x.shape))
f.close()

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

# Base Model
model = CNN_2Stream(class_num)
# optimizer_fn = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=1e-4)
optimizer_fn = tf.optimizers.Adam(learning_rate=1e-4)
# optimizer_fn = tf.optimizers.Adam()
# optimizer_fn = tf.optimizers.SGD()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
f1_score = tfa.metrics.F1Score(class_num, average='weighted')

model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=f1_score)
hist = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y))

model.save_weights(checkpoint_path)

result = f1_score.result()
print("weighted F1 Score : ", result.numpy())

# Convert the model
save_model(model, save_dir + "/base")
converter = tf.lite.TFLiteConverter.from_saved_model(save_dir + "/base")
tflite_model = converter.convert()

# Save the model.
with open(save_dir + '/model_base.tflite', 'wb') as f:
    f.write(tflite_model)
print("Convert Base Model")

#-------------------------------------------------------------------------
# Head Model / include_top=False
model_head = CNN_2Stream(class_num, include_top=False)
model_head(tf.keras.Input(shape=(250, 6)))
model_head.load_weights(checkpoint_path)
# model_head.summary()
# dp = model_head.get_output_shape_at(0)[-1]
# print(dp)

# model_test = Sequential([
#     model_head,
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.05),
#     Dense(class_num, activation='softmax')
# ])
# model_test.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=f1_score)
# hist = model_test.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y))
# model_test.summary()

# Convert the model
save_model(model_head, save_dir + "/head")
converter = tf.lite.TFLiteConverter.from_saved_model(save_dir + "/head")
tflite_model = converter.convert()

# Save the model.
with open(save_dir + '/model_head.tflite', 'wb') as f:
    f.write(tflite_model)
print("Convert Head Model")

#-------------------------------------------------------------------------
# Model Initialize / head model last layer shape = (b, 128)
# head = SoftmaxClassifierHead(batch_size, (128), class_num, l2_reg=1e-4)
head = SoftmaxClassifierHead(batch_size, (9, 1024), class_num)

converter = tf.lite.TFLiteConverter.from_concrete_functions([head.generate_initial_params().get_concrete_function()])
tflite_model = converter.convert()

# Save the model.
with open(save_dir + '/model_init.tflite', 'wb') as f:
    f.write(tflite_model)
print("Convert Init Model")

#-------------------------------------------------------------------------
# Model Train
with tf.Graph().as_default(), tfv1.Session() as sess:
    layer_shape = ((batch_size,) + head.input_shape())
    head_layer = tfv1.placeholder(tf.float32, layer_shape, 'placeholder_bottleneck')
    # One-hot ground truth
    labels = tfv1.placeholder(tf.float32, (batch_size, class_num), 'placeholder_labels')

    loss, gradients, variables = head.train(head_layer, labels)
    converter = tfv1.lite.TFLiteConverter.from_session(sess, [head_layer, labels] + variables, [loss] + gradients)

    if head.train_requires_flex():
        converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    
    tflite_model = converter.convert()

with open(save_dir + '/model_train.tflite', 'wb') as f:
    f.write(tflite_model)
print("Convert Train Model")

#-------------------------------------------------------------------------
# Fully-Connected
with tf.Graph().as_default(), tfv1.Session() as sess:
    layer_shape = ((1,) + head.input_shape())
    head_layer = tfv1.placeholder(tf.float32, layer_shape, 'placeholder_bottleneck')
    predictions, head_variables = head.predict(head_layer)
    # head_layer -> Conv3 shape : [1, 128]
    # head_variables -> fully-connected layer : [128, 8], [8]
    # predictions -> classification : [1, 8]
    converter = tfv1.lite.TFLiteConverter.from_session(sess, [head_layer] + head_variables, [predictions])

    tflite_model = converter.convert()

with open(save_dir + '/model_fc.tflite', 'wb') as f:
    f.write(tflite_model)
print("Convert FC Model")

#-------------------------------------------------------------------------
# Optimizer
def read_parameter_shapes(inference_model):
    """Infers shapes of model parameters from the inference model."""
    interpreter = tfv1.lite.Interpreter(model_content=inference_model)
    return [
        parameter_in['shape'].tolist()
        for parameter_in in interpreter.get_input_details()[1:]
    ]

parameter_shapes = read_parameter_shapes(tflite_model)

optimizer = Adam(learning_rate=1e-4)
# optimizer = SGD(learning_rate=0.01)
optimizer_model = optimizer.generate_optimizer_model(parameter_shapes)

with open(save_dir + '/optimizer.tflite', 'wb') as f:
    f.write(optimizer_model)
print("Convert Optimizer")
