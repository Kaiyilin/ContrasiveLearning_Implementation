# DenseNet
from functions.All_functions import *
import tensorflow.keras.backend as K

class DenseNet3Dbuilder(object):
    """ Building DenseNet"""

    @staticmethod
    def build(input_shape, n_classes, growth_rate, repetitions,  bottleneck_ratio , reg_factor):
        #batch norm + relu + conv
        def bn_rl_conv(x,filters,kernel=(1,1,1),strides=1):
            
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv3D(filters, kernel, strides=strides,padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=reg_factor))(x)
            return x
        
        def dense_block(x, growth_rate, repetition):
            
            for _ in range(repetition):
                y = bn_rl_conv(x, growth_rate*bottleneck_ratio)
                y = bn_rl_conv(y, growth_rate, (3,3,3))
                x = tf.keras.layers.concatenate([y,x])
            return x
            
        def transition_layer(x):
            
            x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
            x = AveragePooling3D((2,2,2), strides = 2, padding = 'same')(x)
            return x
        
        input = Input (input_shape)
        x = Conv3D(96, (7,7,7), strides = 2, padding = 'same')(input)
        x = MaxPooling3D((3,3,3), strides = 2, padding = 'same')(x)
        
        for repetition in repetitions:
            
            d = dense_block(x, growth_rate, repetition)
            x = transition_layer(d)
        x = GlobalAveragePooling3D()(d)
        output = Dense(n_classes, activation = 'softmax')(x)
        
        model = Model(input, output)
        return model

    @staticmethod
    def build_densenet_121(input_shape, n_classes, growth_rate, bottleneck_ratio = 4, reg_factor = 1e-4):
        return DenseNet3Dbuilder.build(input_shape, n_classes, growth_rate, [6, 12, 24, 16], bottleneck_ratio, reg_factor = reg_factor)

    @staticmethod
    def build_densenet_169(input_shape, n_classes, growth_rate, bottleneck_ratio = 4, reg_factor = 1e-4):
        return DenseNet3Dbuilder.build(input_shape, n_classes, growth_rate, [6, 12, 32, 32], bottleneck_ratio, reg_factor = reg_factor)

    @staticmethod
    def build_densenet_201(input_shape, n_classes, growth_rate, bottleneck_ratio = 4, reg_factor = 1e-4):
        return DenseNet3Dbuilder.build(input_shape, n_classes, growth_rate, [6, 12, 48, 32], bottleneck_ratio, reg_factor = reg_factor)
    

    @staticmethod
    def build_densenet_264(input_shape, n_classes, growth_rate, bottleneck_ratio = 4, reg_factor = 1e-4):
        return DenseNet3Dbuilder.build(input_shape, n_classes, growth_rate, [6, 12, 64, 48], bottleneck_ratio, reg_factor = reg_factor)