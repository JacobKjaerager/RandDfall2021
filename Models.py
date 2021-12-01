#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""


import Layers
import keras


def BL(template, dropout=0.1, regularizer=None, constraint=None):
    """
    Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975
    
    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    regularizer: keras regularizer object
    constraint: keras constraint object
    
    outputs
    ------
    keras model object
    """

    inputs = keras.layers.Input(template[0])
    
    x = inputs
    for k in range(1, len(template)-1):
        x = Layers.BL(template[k], regularizer, constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)
    
    x = Layers.BL(template[-1], regularizer, constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs = outputs)
    
    return model

def TABL(template=[[100,40], [60,10], [120,5], [3,1]], dropout=0.1, projection_regularizer=None, projection_constraint=None,
         attention_regularizer=None, attention_constraint=None):
    """
    Temporal Attention augmented Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975
    
    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    projection_regularizer: keras regularizer object for projection matrices
    projection_constraint: keras constraint object for projection matrices
    attention_regularizer: keras regularizer object for attention matrices
    attention_constraint: keras constraint object for attention matrices
    
    outputs
    ------
    keras model object
    """
    
    inputs = keras.layers.Input(template[0])
    
    x = inputs
    for k in range(1, len(template)-1):
        x = Layers.BL(template[k], projection_regularizer, projection_constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)
    
    x = Layers.TABL(template[-1], projection_regularizer, projection_constraint,
                  attention_regularizer, attention_constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs = outputs)
    
    return model

def DeepLOB(lookback_timestep=100, feature_num=40, conv_filter_num=16, inception_num=32,
            LSTM_num=64, leaky_relu_alpha=0.01):
    """
    DeepLOB: Deep Convolutional Neural Networks for Limit Order Books, refer to the paper https://arxiv.org/abs/1808.03668
    All generic values are the chosen values in the article
    
    inputs
    ----
    lookback_timestep: how many timesteps the model will take in, this is also part of the input dimension
    feature_num: how many features the model will use, this is also part of the input dimension
    conv_filter_num: number of convolutional filters in the convolutional blocks of the model
    inception_num: number of inception filters in the inception block of the model
    LSTM_num: number of LSTM filters
    leaky_relu_alpha: value for the alpha in the leaky ReLU layers
    
    outputs
    ------
    keras model object
    """
    
    input_tensor = keras.layers.Input(shape=(lookback_timestep, feature_num, 1))
    
    # Conv block1
    conv_layer1 = keras.layers.Conv2D(conv_filter_num, (1,2), strides=(1, 2))(input_tensor)
    conv_layer1 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer1)
    conv_layer1 = keras.layers.Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer1)
    conv_first1 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer1)
    conv_layer1 = keras.layers.Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer1)
    conv_layer1 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer1)

    # Conv block2
    conv_layer2 = keras.layers.Conv2D(conv_filter_num, (1,2), strides=(1, 2))(conv_layer1)
    conv_layer2 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer2)
    conv_layer2 = keras.layers.Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer2)
    conv_layer2 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer2)
    conv_layer2 = keras.layers.Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer2)
    conv_layer2 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer2)

    # Conv block3
    conv_layer3 = keras.layers.Conv2D(conv_filter_num, (1,10))(conv_layer2)
    conv_layer3 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer3)
    conv_layer3 = keras.layers.Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer3)
    conv_layer3 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer3)
    conv_layer3 = keras.layers.Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer3)
    conv_layer3 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(conv_layer3)
    
    # Inception module
    inception_module1 = keras.layers.Conv2D(inception_num, (1,1), padding='same')(conv_layer3)
    inception_module1 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(inception_module1)
    inception_module1 = keras.layers.Conv2D(inception_num, (3,1), padding='same')(inception_module1)
    inception_module1 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(inception_module1)

    inception_module2 = keras.layers.Conv2D(inception_num, (1,1), padding='same')(conv_layer3)
    inception_module2 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(inception_module2)
    inception_module2 = keras.layers.Conv2D(inception_num, (5,1), padding='same')(inception_module2)
    inception_module2 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(inception_module2)

    inception_module3 = keras.layers.MaxPooling2D((3,1), strides=(1,1), padding='same')(conv_layer3)
    inception_module3 = keras.layers.Conv2D(inception_num, (1,1), padding='same')(inception_module3)
    inception_module3 = keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(inception_module3)
    
    inception_module_final = keras.layers.concatenate([inception_module1, inception_module2, inception_module3], axis=3)
    inception_module_final = keras.layers.Reshape((inception_module_final.shape[1], inception_module_final.shape[3]))(inception_module_final)

    # LSTM
    LSTM_output = keras.layers.LSTM(LSTM_num)(inception_module_final)

    # Fully Connected Layer with softmax activation function for output
    model_output = keras.layers.Dense(3, activation='softmax')(LSTM_output)
    
    DeepLOB_model = keras.Model(inputs=input_tensor, outputs= model_output)  

    return DeepLOB_model
        
    
    