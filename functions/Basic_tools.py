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

from tensorflow.keras.layers import (Input, 
                                    Dense, 
                                    Dropout, 
                                    Flatten, 
                                    Activation, 
                                    BatchNormalization)

from tensorflow.keras.layers import (Conv3D, 
                                     MaxPooling3D, 
                                     Dropout, 
                                     AveragePooling3D, 
                                     GlobalAveragePooling3D, 
                                    )

print('\nImport completed')


# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

# Read files
def myreadfile(dirr):
    """
    This version can import 3D array regardless of the size
    """
    os.chdir(dirr)
    #cwd = os.getcwd()

    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort() #對讀取的路徑進行排序
    for file in path_list:
          if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def padding_zeros(array, pad_size):
    # define padding size
    elements = array.shape    
    for element in elements:
        if element > pad_size:
            sys.exit('\nThe expanded dimension shall be greater than your current dimension')
    pad_list = list() 
    for i in range(array.ndim):
        x = pad_size - array.shape[i]
        if x%2 ==1:
            y_1 = (x/2 +0.5)
            y_2 = (x/2 -0.5)
            z = (int(y_1),int(y_2))
            pad_list.append(z)

        else:
            y = int(x/2)
            z=(y,y)
            pad_list.append(z)
    pad_array = np.pad(array, pad_list, 'constant')
    pad_list = list() 
    return pad_array

def myreadfile_pad(dirr, pad_size):
    
    #This version can import 3D array regardless of the size
    
    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def myreadfile_resample_pad(dirr, pad_size):
    #This version can import 3D array regardless of the size
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    template = load_mni152_template()

    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = resample_to_img(img, template)
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def importdata_resample(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):
    def myreadfile_resample_pad(dirr, pad_size):
        #This version can import 3D array regardless of the size
        from nilearn.datasets import load_mni152_template
        from nilearn.image import resample_to_img
        template = load_mni152_template()

        os.chdir(dirr)
        number = 0

        flag = True
        imgs_array = np.array([])
        path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
        path_list.sort()
        for file in path_list:
            if file.endswith(".nii"):
                #print(os.path.join(dirr, file))
                img = nib.load(os.path.join(dirr, file))
                img_array = resample_to_img(img, template)
                img_array = img.get_fdata()
                img_array = data_preprocessing(img_array)
                img_array = padding_zeros(img_array, pad_size)
                img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
                number += 1
                if flag == True:
                    imgs_array = img_array

                else:
                    imgs_array = np.concatenate((imgs_array, img_array), axis=0)

                flag = False
        return number, imgs_array, path_list
    if pad_size == None:
      _, first_mo,  = myreadfile(dirr)
      _, second_mo, _ = myreadfile(dirr1)
      _, third_mo, _ = myreadfile(dirr2)
      
      _, first_mo2, _ = myreadfile(dirr3)
      _, second_mo2, _ = myreadfile(dirr4)
      _, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      _, first_mo, _ = myreadfile_resample_pad(dirr,pad_size)
      _, second_mo, _ = myreadfile_resample_pad(dirr1,pad_size)
      _, third_mo, _ = myreadfile_resample_pad(dirr2,pad_size)
      
      _, first_mo2, _ = myreadfile_resample_pad(dirr3,pad_size)
      _, second_mo2, _ = myreadfile_resample_pad(dirr4,pad_size)
      _, third_mo2, _ = myreadfile_resample_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

def importdata(dirr,dirr1,dirr2,dirr3,dirr4,dirr5):
    
    a_num, first_mo, _ = myreadfile(dirr)
    b_num, second_mo, _ = myreadfile(dirr1)
    h_num, third_mo, _ = myreadfile(dirr2)
    
    a_num2, first_mo2, _ = myreadfile(dirr3)
    b_num2, second_mo2, _ = myreadfile(dirr4)
    h_num2, third_mo2, _ = myreadfile(dirr5)
    print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
    return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

def importdata2(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):

    if pad_size == None:
      a_num, first_mo, _ = myreadfile(dirr)
      b_num, second_mo, _ = myreadfile(dirr1)
      h_num, third_mo, _ = myreadfile(dirr2)
      
      a_num2, first_mo2, _ = myreadfile(dirr3)
      b_num2, second_mo2, _ = myreadfile(dirr4)
      h_num2, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      #pad_size = int(input('Which size would you like? '))
      a_num, first_mo, _ = myreadfile_pad(dirr,pad_size)
      b_num, second_mo, _ = myreadfile_pad(dirr1,pad_size)
      h_num, third_mo, _ = myreadfile_pad(dirr2,pad_size)
      
      a_num2, first_mo2, _ = myreadfile_pad(dirr3,pad_size)
      b_num2, second_mo2, _ = myreadfile_pad(dirr4,pad_size)
      h_num2, third_mo2, _ = myreadfile_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2


def split(c,array):
    array_val = array[:c,:,:,:]
    array_tr = array[c:,:,:,:]
    return array_tr, array_val

# Basic conv block 
def bn_relu_block(input,filter,kernel_size,param):
    y = Conv3D(filter, kernel_size, padding='same', use_bias=True,kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(input)
    y = BatchNormalization()(y)
    act = Activation('relu')(y)
    tf.summary.histogram("Activation", y)# testing 
    return y

# Basic SE, inception and parametre reduction module

def half_reduction(input,filters):
    
    conv00 = Conv3D(filters,(1,1,1),padding='same')(input)
    conv00 = Conv3D(filters,(3,3,3),padding='same')(conv00)
    conv00 = Conv3D(filters, (3,3,3),strides=(2,2,2),padding='same')(conv00)

    conv01 = Conv3D(filters,(1,1,1),padding='same')(input)
    conv01 = Conv3D(filters, (3,3,3),strides=(2,2,2),padding='same')(conv01)
 
    avg00 = MaxPooling3D(pool_size=(2,2,2))(input)

    concatenate = tf.keras.layers.concatenate([conv00, conv01, avg00])
    concatenate = BatchNormalization()(concatenate)
    #y = Activation('relu')(y)
    return Activation('relu')(concatenate)

def inception_module3D(input_img,filters,param):
    incep_1 = Conv3D(filters, (1,1,1), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input_img)
    incep_1 = Conv3D(filters, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(incep_1)
    incep_2 = Conv3D(filters, (1,1,1), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input_img)
    incep_2 = Conv3D(filters, (5,5,5), padding='same', activation='relu')(incep_2)
    incep_3 = MaxPooling3D((3,3,3), strides=(1,1,1), padding='same')(input_img)
    incep_3 = Conv3D(filters, (1,1,1), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(incep_3)
    output = tf.keras.layers.concatenate([incep_1, incep_2, incep_3], axis = 4)
    return output
    
def inc_module_A(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (3,3,3),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  return output

def inc_module_B(input,filters,param):

  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  
  chan_2 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (7,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_2 = Conv3D(filters, (1,7,7),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  return output

def inc_module_C(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_12 = Conv3D(filters, (3,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_13 = Conv3D(filters, (1,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_21 = Conv3D(filters, (1,3,3),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_22 = Conv3D(filters, (3,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_12, chan_13, chan_21, chan_22, chan_3, chan_4],axis=4)
  return output
    
def inc_module_A_2(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (3,3,3), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1), kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  return output

def inc_module_B_2(input,filters,param):

  chan_1 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)


  chan_2 = Conv3D(filters, (1,1,1),padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (7,1,1),padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_2 = Conv3D(filters, (1,7,7),padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1), kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  return output

def inc_module_C_2(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_12 = Conv3D(filters, (3,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_13 = Conv3D(filters, (1,3,3), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_21 = Conv3D(filters, (1,3,3), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_22 = Conv3D(filters, (3,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1), kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_12, chan_13, chan_21, chan_22, chan_3, chan_4],axis=-1)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  return output

def se_block_3D(tensor, ratio):
    nb_channel = tensor.shape[-1] # for channel last

    x = GlobalAveragePooling3D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu',use_bias=False)(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = tf.keras.layers.Multiply()([tensor, x])
    return x

def Conv_SE_block(input, filters, kernel_size, param, ratio, SE = True):
    """
    build a conv --> SE block
    """
    y = Conv3D(filters, 
               kernel_size, 
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=0,l2=param), 
               bias_regularizer=l1_l2(l1=0,l2=0))(input)
    if SE == True:
        y = se_block_3D(y, ratio)
    else: 
        y=y
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def create_model():
    input1 = Input(shape=(53,63,52,1))
    conv11 = Conv3D(128, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.0002))(input1)
    conv12 = Conv3D(64, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.0002))(conv11)
    pool12 = MaxPooling3D(pool_size=(2, 2, 2))(conv12)
    conv13 = Conv3D(32, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.0002))(pool12)
    conv14 = Conv3D(16, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.0002))(conv13)
    pool13 = MaxPooling3D(pool_size=(2, 2, 2))(conv14)
    inc = inception_module3D(pool13,32)
    flat1 = Flatten()(inc)

    # interpretation model
    hidden1 = Dense(64, activation='relu')(flat1)
    hidden2 = Dense(64, activation='relu')(hidden1)
    drop1 = Dropout(0.5)(hidden2)
    hidden3 = Dense(32, activation='relu')(hidden2)
    drop2 = Dropout(0.4)(hidden3)
    hidden4 = Dense(16, activation='relu')(drop2)
    hidden5 = Dense(8, activation='relu')(hidden4)
    drop3 = Dropout(0.2)(hidden3)
    output = Dense(2, activation='softmax')(hidden5)
    model = Model(inputs=input1, outputs=output)
    print(model.summary())
    return model    


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