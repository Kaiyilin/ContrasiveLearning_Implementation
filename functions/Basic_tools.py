 # Multiple Inputs
import os
import sys
import cv2
import scipy
import datetime
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2, l1, l1_l2 # By using both, you can implemented the concept of elastic dense net
print('\nImport completed')


# model training

def base_model_creator(model, train_para = False):
    base_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-2).output])
    base_model.trainable = train_para
    return base_model

def model_structure(model):
    """
    Visualise model's architecture
    display feature map shapes
    """
    for i in range(len(model.layers)):
      layer = model.layers[i]
    # summarize output shape
      print(i, layer.name, layer.output.shape)

def decay(epoch):
  if epoch <= 30:
    return 1e-1
  elif epoch > 30 and epoch <= 70:
    return 1e-2
  else:
    return 1e-3

# Setting callbacks

# tensorboard log directiory
logdir="/home/kaiyi/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# model checkpoint
checkpoint_dir ="/home/kaiyi/trckpt/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))



print(checkpoint_dir)
print(logdir)

print("\ntf.__version__ is", tf.__version__)
print("\ntf.keras.__version__ is:", tf.keras.__version__)