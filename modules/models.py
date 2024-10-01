#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:50:13 2020

@author: filippomartinini
"""

# third party
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Activation, LeakyReLU, Permute
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Multiply
from tensorflow.keras.layers import BatchNormalization, Concatenate, Add
from tensorflow.keras.layers import Subtract, Dense, Flatten
from tensorflow.keras.initializers import Initializer
from tensorflow import keras
from modules import layers
import copy


############# LOUPE evolutions #############

    
def dec2(input_shape,
         R,
         depth = 5,
         name = None,
         cartesian_type = None,
         fixed_mask_from_outside = False,
         mask = None,
        ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)
    
    if fixed_mask_from_outside == False:
        last_tensor_mask = _mask_from_tensor(last_tensor, R, cartesian_type = cartesian_type)
        
    else:
        assert mask is not None, 'provide a mask'
        last_tensor_mask = layers.FixedMask(mask)(last_tensor_Fx)

    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx,
                                                        last_tensor_mask])
    
    last_tensor = layers.IFFT(name='ifft')(y)
        
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth)

    last_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
    
    last_tensor = layers.FFT(name='fft_projection')(last_tensor)
         
    last_tensor = layers.UnderSampleHolistic(
            complement = 1,                                 
            name='undersample_projection_complement',)([last_tensor,
                                                        last_tensor_mask])
    
    last_tensor = Add(name='add_projection')([last_tensor, y])
    
    last_tensor = layers.IFFT(name='ifft_projection')(last_tensor)
    
    last_tensor = layers.ComplexAbs(name='abs_projection')(last_tensor)
    
    if name is None:
        name = 'dec2'
    else:
        name = name
        
    return Model(inputs = inputs, outputs = last_tensor, name = name)


def dec1(input_shape,
         R,
         L = 2,
         depth = 5,
         name = None,
        ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    assert L==1 or L==2, 'only "L==1" or "L==2" are valid entries.'
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)

    last_tensor_mask = _mask_from_tensor(last_tensor, R,)
    
    # Under-sample with the mask or the binary version of the mask
    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx, 
                                                        last_tensor_mask])
                                   
    if L == 2:
        y_bar = layers.UnderSampleHolistic(complement = 1,
                                           name='undersample_complement',
                                          )([last_tensor_Fx, 
                                             last_tensor_mask])

    # IFFT if trainTIFT==False, TIFT if trainTIFT==True
    last_tensor = layers.IFFT(name='ifft')(y)
        
    # hard-coded UNet
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth, )

    # final output from model 
    add_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
    
    # complex absolute layer
    abs_tensor = layers.ComplexAbs(name='abs')(add_tensor)
    
    last_tensor = layers.FFT(name='fft_projection')(add_tensor)
    
    
    
    
    y_preojected_for_regularization = layers.UnderSampleHolistic(
            complement = L-1,                                                      
            name='undersample_projection')([last_tensor, last_tensor_mask])
    
    if L == 1:
        y_for_regularization = y
    elif L == 2:
        y_for_regularization = y_bar
        
    regularization = Subtract(name='projection')([
            y_preojected_for_regularization, 
            y_for_regularization])
    
    
    outputs = [abs_tensor, regularization]
    
    if name is None:
        name = 'dec1-L'+str(L)
    else:
        name = name
    
    return Model(inputs = inputs, outputs = outputs, name = name)

def dec0(input_shape,
         R,
         depth = 5,
         name = None,
        ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)

    last_tensor_mask = _mask_from_tensor(last_tensor, R,)
    
    # Under-sample with the mask or the binary version of the mask
    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx, 
                                                        last_tensor_mask])
                                   
    # IFFT if trainTIFT==False, TIFT if trainTIFT==True
    last_tensor = layers.IFFT(name='ifft')(y)
        
    # hard-coded UNet
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth, 
                                    output_nb_feats = 1)
    
    # complex absolute layer
    last_tensor = layers.ComplexAbs(name='abs')(last_tensor)
    
    # final output from model 
    last_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
    
    if name is None:
        name = 'dec0'
    else:
        name = name
        
    return Model(inputs = inputs, outputs = last_tensor, name = name)


############# Trainable Mask #############

def _mask_from_tensor(last_tensor, R, pmask_slope=5, sample_slope=200, cartesian_type = None, ):
    
    """
    Mask used in LOUPE
    """
    
    # build probability mask
    prob_mask_tensor = layers.ProbMask(
        name='prob_mask',
        slope=pmask_slope,
        cartesian_type = cartesian_type)(last_tensor) 

    # probability mask rescaled to have mean=sparsity
    prob_mask_tensor_rescaled = layers.RescaleProbMap(R, 
                                                      name='prob_mask_scaled',
                                                     )(prob_mask_tensor)

    # Realization of random uniform mask
    thresh_tensor = layers.RandomMask(
        name='random_mask', 
        cartesian_type = cartesian_type,
    )(prob_mask_tensor)

    # Realization of mask
    last_tensor_mask = layers.ThresholdRandomMask(slope=sample_slope, 
                                                  name='sampled_mask',
                                                 )([prob_mask_tensor_rescaled,
                                                    thresh_tensor]) 

    return last_tensor_mask
        
    
############# U-NET #############


def _unet_from_tensor(tensor, 
                      filt = 64, 
                      kern = 3, 
                      depth = 5, 
                      trainable = True, 
                      acti = None, 
                      output_nb_feats = 2, 
                      batch_norm_before_acti = True,
                      pool_size = (2, 2),
                     ):

    output_tensor = tensor
    tensor_list = []
    for i in np.arange(1, depth):
        tensor = basic_UNET_block(
                output_tensor, filt*(2**(i-1)),
                kern, acti, i, trainable, 
                batch_norm_before_acti = batch_norm_before_acti)

        output_tensor = AveragePooling2D(pool_size=pool_size,
                                         name = 'pool_'+str(i), )(tensor)

        tensor_list += [tensor]

    output_tensor = basic_UNET_block(
            output_tensor, filt*(2**(depth-1)), 
            kern, acti, depth, trainable, 
            batch_norm_before_acti=batch_norm_before_acti)

    tensor_list = tensor_list[::-1]

    for j, i in enumerate(np.arange(depth+1, 2*depth)):

        output_tensor = UpSampling2D(size=pool_size, name = 'up_'+str(j),
                                    )(output_tensor)

        output_tensor = Concatenate(axis=-1, name = 'concat_'+str(j), 
                                   )([output_tensor, tensor_list[j]])

        output_tensor = basic_UNET_block(
                output_tensor, 
                filt*(2**(depth-2-j)), kern, acti, i, trainable,
                batch_norm_before_acti=batch_norm_before_acti)

    output_tensor = Conv2D(output_nb_feats, 1, padding = 'same',
                           name='output_UNET', 
                           trainable = trainable)(output_tensor)

    return output_tensor


def basic_UNET_block(inp, filt, kern, acti, identifier,
                     trainable = True, 
                     batch_norm_before_acti=False):
            
    idf = str(identifier)

    conv = Conv2D(filt, kern, activation = acti, padding = 'same',
                  name='conv_'+idf+'_1', trainable = trainable)(inp)
    conv = batch_norm_and_relu(conv, idf, '1', 
                               batch_norm_before_acti=batch_norm_before_acti,)
    conv = Conv2D(filt, kern, activation = acti, padding = 'same',
                  name='conv_'+idf+'_2', trainable = trainable)(conv)
    conv = batch_norm_and_relu(conv, idf, '2',
                               batch_norm_before_acti=batch_norm_before_acti,)

    return conv
        
def batch_norm_and_relu(conv, idf, idf_2, batch_norm_before_acti=False,):
    ReLu = LeakyReLU(name = 'leaky_re_lu_'+idf+'_'+idf_2)

    if batch_norm_before_acti == False:
        conv = ReLu(conv)

    conv = BatchNormalization(name='batch_norm_'+idf+'_'+idf_2)(conv)

    if batch_norm_before_acti == True:
        conv = ReLu(conv)

    return conv

############# Dykstra #############

def add_Dykstra_projection_to_model(model, iterations = 15, name = None):
    
    y = model.get_layer('undersample').output
    
    mask = model.get_layer('sampled_mask').output
    
    last_tensor = model.outputs[0]
    
    for i in range(iterations):
        
        last_tensor = layers.ConcatenateZero(name='concat_zero_Dykstra'+str(i),
                                            )(last_tensor)         
        
        last_tensor = layers.FFT(name='fft_Dykstra-'+str(i))(last_tensor)
        
        last_tensor = layers.UnderSampleHolistic(
                complement = True, 
                 hard_threshold = True, 
                 name='undersample_Dykstra'+str(i),
                )([last_tensor, mask])
        
        last_tensor = Add(name='add_Dykstra-'+str(i))([last_tensor, y])
        
        last_tensor = layers.IFFT(name='ifft_Dykstra-'+str(i))(last_tensor)
        
        last_tensor = layers.ComplexAbs(name='abs_Dykstra-'+str(i))(last_tensor)
    
        last_tensor = layers.Clip(name='clip_Dykstra-'+str(i))(last_tensor)
    
    
    inputs = model.inputs
    outputs = last_tensor
    
    if name is None:
        name = model.name+'-Dykstra'
    else:
        name = name
    
    model_Dykstra = Model(inputs = inputs, outputs = outputs, 
                          name = name)
    return model_Dykstra

def add_Dykstra_Functional_projection_to_model(model, iterations = 15, name = None):
    
    y = model.get_layer('undersample').output
    
    mask = model.get_layer('sampled_mask').output
    
    x = model.outputs[0]
    
    input_last_tensor = tf.keras.Input(x.shape[1:])
    input_mask = tf.keras.Input(mask.shape[1:])
    input_y = tf.keras.Input(y.shape[1:])
    
    inputs = [input_last_tensor, input_mask, input_y]
    
    for i in range(iterations):
        
        if i == 0:
            last_tensor = layers.ConcatenateZero(name='concat_zero_Dykstra'+str(i),
                                        )(input_last_tensor)
        else:
            last_tensor = layers.ConcatenateZero(name='concat_zero_Dykstra'+str(i),
                                                )(last_tensor)         
        
        last_tensor = layers.FFT(name='fft_Dykstra-'+str(i))(last_tensor)
        
        last_tensor = layers.UnderSampleHolistic(
                complement = True, 
                 hard_threshold = True, 
                 name='undersample_Dykstra'+str(i),
                )([last_tensor, input_mask])
        
        last_tensor = Add(name='add_Dykstra-'+str(i))([last_tensor, input_y])
        
        last_tensor = layers.IFFT(name='ifft_Dykstra-'+str(i))(last_tensor)
        
        last_tensor = layers.ComplexAbs(name='abs_Dykstra-'+str(i))(last_tensor)
    
        last_tensor = layers.Clip(name='clip_Dykstra-'+str(i))(last_tensor)
    
    
    outputs = last_tensor
    model_Dykstra = tf.keras.Model(inputs, outputs, name = 'Dykstra_functional')
    
    inputs = model.inputs
    last_tensor = model.outputs
    outputs = model_Dykstra([x, mask, y])
    
    if name is None:
        name = model.name+'-Dykstra'
    else:
        name = name
        
    return tf.keras.Model(inputs, outputs, name = name)


############# Masker #############

def add_masker(model, masker = None, name = None):
    
    model_before_masker = Model(model.inputs,
                                [model.get_layer('fft').output,
                                 model.get_layer('sampled_mask').output],
                                name = 'encoder'
                               )
    
    
    [last_tensor, last_tensor_mask] = model_before_masker.outputs
    
    last_tensor_mask = layers.Masker(masker = masker)(last_tensor_mask)
    
    y = model.get_layer('undersample')([last_tensor, last_tensor_mask])
    
    last_tensor = Model(model.get_layer('ifft').input,
                        model.get_layer('fft_projection').output,
                        name = 'decoder',
                       )(y)
    
    y_bar_hat = model.get_layer('undersample_projection_complement')([last_tensor,
                                                                      last_tensor_mask])
    
    outputs = Model(model.get_layer('add_projection').input,
                    model.outputs,
                    name = 'projection',
                   )([y_bar_hat, y])
    
    if name is None:
        name = 'model_masker'
    else:
        name = name
    
    model_masker = Model(inputs = model_before_masker.inputs, 
                         outputs = outputs, 
                         name = name, )
        
    return model_masker
    
############# Other models #############

def model_Fourier(input_shape, mode = 'direct'):
    
    assert mode == 'direct' or model == 'inverse', '"mode" is "direct" or "inverse"'
    
    input_x = Input(shape=input_shape, name='input_x')
    
    last_tensor = input_x
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero',
                                            )(last_tensor)
        pass
    
    if mode == 'inverse':
        last_tensor = layers.FFT()(last_tensor)
    else:
        last_tensor = layers.IFFT()(last_tensor)
        
    inputs = [input_x]
    outputs = [last_tensor]
    modelFourier = Model(inputs=inputs,outputs=outputs, 
                         name = mode+'_fast_fourier_transform')
    return modelFourier

def encoder(model, ):
    inputs = model.inputs
    last_tensor = model.get_layer('ifft').output
    last_tensor = layers.ComplexAbs(name='abs')(last_tensor)
    encoder = Model(inputs = inputs, outputs = last_tensor)
    return encoder

def model_sub_results_for_self_assessment(model_Dykstra, version = 1):
    
    fft_x_bar = model_Dykstra.get_layer('decoder').get_layer('fft_projection').output
    last_tensor_mask = model_Dykstra.get_layer('masker').output


    y_hat = layers.UnderSampleHolistic(
        name='undersample_sub_results',)([fft_x_bar,
                                          last_tensor_mask])

    y = model_Dykstra.get_layer('undersample').output

    y_diff = tf.keras.layers.Subtract(name = 'subtract_sub_results_y')([y, y_hat])

    x = model_Dykstra.inputs
    x_hat = model_Dykstra.outputs

    x_diff = tf.keras.layers.Subtract(name = 'subtract_sub_results_x')([x[0], x_hat[0]])
    
    if version == 1:
        model_sub_results = Model(
            x,
            [y_diff,
             x_diff,
             x_hat,
            ]
        )
        
    elif version == 2:
        
        Finv_y = layers.IFFT(name = 'ifft_y_sub')(y)

        Finv_y_hat = layers.IFFT(name = 'ifft_y_hat_sub')(y_hat)    
    
        model_sub_results = Model(
                x,
                [y_diff,
                 Finv_y,
                 Finv_y_hat,
                 x_diff,
                 x_hat,
                ]
            )
    
    return model_sub_results




############# Self-Assessment #############

def model_self_assessment(input_shape_image, 
                          input_shape_metric,
                          line_coeff = [0, 1, 0], 
                          filt = [10,8,6,4,2,1],
                          kern = [5,5,5,5,5,5],
                          pool_size = [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2)],
                          average_pooling = True,
                          poly_degree = 2,
                          dense = False,
                          conv_per_block_image = 1,
                          name = None,
                          last_activation = None,
                         ):
    
    if last_activation is None:
        last_activation = 'linear'
    
    inputs = [Input(shape=input_shape_image, name='input_image'), 
              Input(shape=input_shape_metric, name='input_metric'),
             ]
    
    last_tensor = layers.PolynomialPower(poly_degree, name = 'poly_power')(inputs[1])
        
    last_tensor = Dense(1,
                        name = 'linear_reg_dense',
                        activation = 'linear',
                        kernel_initializer = layers._InitializerLinRegDense(line_coeff[:-1]),
                        bias_initializer = tf.keras.initializers.Constant(value=line_coeff[-1]),
                        trainable = True, 
                       )(last_tensor)

    conv = inputs[0]
    n = len(filt)-1
    
    for ind_block, (f, k, p) in enumerate(zip(filt, kern, pool_size)):
        
        for j in range(conv_per_block_image):
            
            if conv_per_block_image == 1:
                ind_conv_block = ''
            else:
                ind_conv_block = '_'+str(j)
            
            ind_layer_tmp = '_'+str(ind_block)+ind_conv_block
            
            conv = Conv2D(f, k, padding = 'same', 
                          name='conv'+ind_layer_tmp)(conv)
            conv = LeakyReLU(name = 'relu'+ind_layer_tmp)(conv)
            conv = BatchNormalization(name='batch_norm'+ind_layer_tmp)(conv)
        
        if p!=(0,0) or ind_block==n:
            if average_pooling==True:
                if ind_block==n:
                    if dense == False:
                        last_tensor_image = GlobalAveragePooling2D(
                            name = 'global_average_pool')(conv)
                    else:
                        flat = Flatten(name = 'flatten')(conv)
                        # last_tensor_image = Dense(100, name = 'dense_image_0')(flat)
                        last_tensor_image = Dense(1, name = 'dense_image_1')(flat)
                else:
                    conv = AveragePooling2D(
                        pool_size=p, name = 'average_pool_'+str(ind_block))(conv)

            else:
                if ind_block==n:
                    if dense == False:
                        last_tensor_image = GlobalMaxPooling2D(
                            name = 'global_max_pool')(conv)
                    else:
                        flat = Flatten(name = 'flatten')(conv)
                        # last_tensor_image = Dense(100, name = 'dense_image_0')(flat)
                        last_tensor_image = Dense(1, name = 'dense_image_1')(flat)

                else:
                    conv = MaxPooling2D(
                        pool_size=p, name = 'max_pool_'+str(ind_block))(conv)
                
  
    concat = Concatenate(axis=-1, name = 'concat')([last_tensor_image, last_tensor])

    last_tensor = Dense(1,
                        name = 'correction',
                        activation=last_activation,
                        kernel_initializer=tf.keras.initializers.Constant(value=[1]),
                        bias_initializer=tf.keras.initializers.Constant(value=[0]),
                        trainable = True, 
                       )(concat)
    
    outputs = [last_tensor]
    
    if name is None:
        name = 'model_self_assessment'
    else:
        name = name
    
    return Model(inputs=inputs,outputs=outputs, name = name)



# Basic ResNet Building Block


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first = True,
                ):
    
    conv=Conv2D(num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer='he_normal',
               )

    x=inputs
    
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

                 
 # ResNet architecture
def resnet(input_shape, 
           depth,
           num_classes=10,
          ):
    
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])')
        
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True,
                    )

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0: # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0: # first layer but not first stage
                    strides = 2 # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(1,
                    activation='linear',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def conv_dense_block(
        input_shape,
        filt,
        kern,
        pool_size,
        units, 
        conv_per_block_image = 1,
        average_pooling = False,
        dense = True,
        id_str = '',
    ):
    
    n = len(filt)-1
    
    inputs = Input(input_shape, name = 'input_'+id_str)
    conv = inputs
    
    for ind_block, (f, k, p) in enumerate(zip(filt, kern, pool_size)):
        
        for j in range(conv_per_block_image):
            
            if conv_per_block_image == 1:
                ind_conv_block = ''
            else:
                ind_conv_block = '_'+str(j)
            
            ind_layer_tmp = '_'+str(ind_block)+ind_conv_block+'_'+id_str
            
            conv = Conv2D(f, k, padding = 'same', 
                          name='conv'+ind_layer_tmp)(conv)
            conv = LeakyReLU(name = 'relu'+ind_layer_tmp)(conv)
            conv = BatchNormalization(name='batch_norm'+ind_layer_tmp)(conv)
        
        if p!=(0,0) or ind_block==n:
            if average_pooling==True:
                if ind_block==n:
                    if dense == False:
                        last_tensor_image = GlobalAveragePooling2D(
                            name = 'global_average_pool')(conv)
                    else:
                        last_tensor_image = Flatten(name = 'flatten'+'_'+id_str)(conv)
                        for j, u in enumerate(units):
                            ind_layer_tmp = '_'+str(j)+'_'+id_str
                            last_tensor_image = Dense(u, name = 'dense_'+ind_layer_tmp)(last_tensor_image)
                            last_tensor_image = LeakyReLU(name = 'dense_relu_'+ind_layer_tmp)(last_tensor_image)
                            last_tensor_image = BatchNormalization(name='dense_batch_norm_'+ind_layer_tmp)(last_tensor_image)
                            
                        last_tensor_image = Dense(1, name = 'dense_image_out_'+id_str)(last_tensor_image)
                else:
                    conv = AveragePooling2D(
                        pool_size=p, name = 'average_pool_'+str(ind_block)+'_'+id_str)(conv)

            else:
                if ind_block==n:
                    if dense == False:
                        last_tensor_image = GlobalMaxPooling2D(
                            name = 'global_max_pool_'+id_str)(conv)
                    else:
                        last_tensor_image = Flatten(name = 'flatten'+'_'+id_str)(conv)
                        for j, u in enumerate(units):
                            ind_layer_tmp = '_'+str(j)+'_'+id_str
                            last_tensor_image = Dense(u, name = 'dense_'+ind_layer_tmp)(last_tensor_image)
                            last_tensor_image = LeakyReLU(name = 'dense_relu_'+ind_layer_tmp)(last_tensor_image)
                            last_tensor_image = BatchNormalization(name='dense_batch_norm_'+ind_layer_tmp)(last_tensor_image)
                            
                        last_tensor_image = Dense(1, name = 'dense_image_out_'+id_str)(last_tensor_image)

                else:
                    conv = MaxPooling2D(
                        pool_size=p, name = 'max_pool_'+str(ind_block)+'_'+id_str)(conv)

    conv_dense_model = tf.keras.Model(
            inputs,
            last_tensor_image,    
            name = 'conv_dense_sub_model_'+id_str,
        )
        
    return conv_dense_model

def model_self_assessment_new(
        input_shape_image, 
        input_shape_metric,
        filt = [10,8,6,4,2,1],
        kern = [5,5,5,5,5,5],
        pool_size = [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2)],
        units = [100, 50,],
        average_pooling = True,
        poly_degree = 2,
        dense = False,
        conv_per_block_image = 1,
        name = None,
    ):
    
    inputs = [
            Input(shape=input_shape_image, name = 'input_image'),
            Input(shape=input_shape_metric, name = 'input_metric'),
        ]
    
    input_x = inputs[0]
    input_y = inputs[1]
        
    last_tensor_metric = conv_dense_block(
            input_shape_metric,
            filt,
            kern,
            pool_size,
            units, 
            conv_per_block_image = conv_per_block_image,
            average_pooling = False,
            dense = True,
            id_str = 'metric',
        )(input_y)
    
    last_tensor_image = conv_dense_block(
            input_shape_image,
            filt,
            kern,
            pool_size,
            units, 
            conv_per_block_image = conv_per_block_image,
            average_pooling = False,
            dense = True,
            id_str = 'image',
        )(input_x)
  
    concat = Concatenate(axis=-1, name = 'concat')([last_tensor_image, last_tensor_metric])

    class Initializer_1_0(tf.keras.initializers.Initializer):
        
        def __call__(self, shape, dtype=None, **kwargs):
            
            z = np.zeros(shape)
            
            z[0] = 1
            
            z = tf.convert_to_tensor(z, dtype=tf.float32)
                        
            return z
            
        def get_config(self):  # To support serialization
            return {}
    
    last_tensor = Dense(1,
                        name = 'correction',
                        activation='linear',
                        kernel_initializer=Initializer_1_0(),
                        # kernel_initializer=tf.keras.initializers.Constant(value=[1]),
                        bias_initializer=tf.keras.initializers.Constant(value=[0]),
                        trainable = True, 
                       )(concat)
    
    outputs = [last_tensor]
    
    if name is None:
        name = 'model_self_assessment'
    else:
        name = name
    
    return Model(inputs=inputs,outputs=outputs, name = name)

# model I2TCM...

def conv_chain(
        last_tensor,
        kernel_size_list,
        filt_list,
        pool_list,
        stride_pool_list,
        dropout_conv = False,
        identifier = '1',
        conv_D = 1,
    ):

    if conv_D == 1:
        conv_layer = tf.keras.layers.Conv1D
        pool_layer = tf.keras.layers.MaxPooling1D
    elif conv_D == 2:
        conv_layer = tf.keras.layers.Conv2D
        pool_layer = tf.keras.layers.MaxPooling2D

    for id_tmp, (kernel_size, filt, pool, stride) in enumerate(
            zip(kernel_size_list, filt_list, pool_list, stride_pool_list)):
        
        last_tensor = conv_layer(
                filters = filt,
                kernel_size = kernel_size,
                strides = 1,
                padding = 'same',
                activation = 'relu',
                name = 'conv_'+identifier+str(id_tmp),
            )(last_tensor)

        last_tensor = tf.keras.layers.BatchNormalization(
                name = 'batch_norm_conv_'+identifier+str(id_tmp),
            )(last_tensor)

        if dropout_conv == True:
            last_tensor = tf.keras.layers.Dropout(
                    rate = 0.05,
                    name = 'dropout_conv_'+identifier+str(id_tmp),
                )(last_tensor)

        if pool != 0:

            last_tensor = pool_layer(
                        pool_size=pool,
                        strides = stride,
                        name = 'pool_'+identifier+str(id_tmp),
                    )(last_tensor)

    return last_tensor

def model_self_assessment_I2TMC(
            input_shape_x_hat,
            input_shape_y,
            input_shape_y_hat,
            target = 'regression',
            name = 'my_DNN',
        ):
    
    """ MODEL TYPE = 5
    xh --- shape=(n,1)
    y  --- shape=(n,1)
    A  --- shape=(n,n,1)
    
    x_c = conv1D(xh)       --- shape=(n,n//2)
    y_c = conv1D(y)        --- shape=(n,n//2)
    xy = concat(x_c, y_c)  --- shape=(n,n,1)
    xyA = concat(xy, A)    --- shape=(n,n,2)
    xyA_conv = conv2D(xyA) --- shape=(n,n,n//2)
    
    output = Dense(flatten(xyA_conv))
    """
    
    n = input_shape_x_hat[0]
    
    input_x_hat = tf.keras.Input(
            shape=input_shape_x_hat, 
            name="input_x_hat",
        )
    
    input_y = tf.keras.Input(
            shape=input_shape_y, 
            name="input_y",
        )
    
    input_y_hat = tf.keras.Input(
            shape=input_shape_y_hat, 
            name="input_y_hat",
        )
    
    kernel_size_list = [3] * 4
    filt_list = [n//(2**(8-i)) for i in range(4)]
    pool_list = [2, 0, 2, 0]
    stride_pool_list = [None] * 4
    
    last_tensor_x_hat = conv_chain(
            input_x_hat,
            kernel_size_list,
            filt_list,
            pool_list,
            stride_pool_list,
            identifier = 'X',
            conv_D = 2,
        )
    
    last_tensor_y = conv_chain(
            input_y,
            kernel_size_list,
            filt_list,
            pool_list,
            stride_pool_list,
            identifier = 'Y',
            conv_D = 2,
        )
    
    last_tensor_y_hat = conv_chain(
            input_y_hat,
            kernel_size_list,
            filt_list,
            pool_list,
            stride_pool_list,
            identifier = 'Y_hat',
            conv_D = 2,
        )
    
    last_tensor = tf.keras.layers.Concatenate(
            axis = -1,
        )([last_tensor_y, last_tensor_y_hat, last_tensor_x_hat, ])
    
    kernel_size_list = [(3, 3)] * 4
    filt_list = [n//(2**(8-i)) for i in range(4)]
    pool_list = [3, 3, 5, 5]
    stride_pool_list = [2] * 4
    
    last_tensor = conv_chain(
            last_tensor,
            kernel_size_list,
            filt_list,
            pool_list,
            stride_pool_list,
            identifier = 'AXY',
            conv_D = 2,
        ) ######################## CHESK POOLI
    
    last_tensor = tf.keras.layers.Flatten(
            name = 'flatten',
        )(last_tensor)
    
    ### DENSE LAYERS
    
    units_list = [50, 50, 25]
        
    for id_tmp, units in enumerate(units_list):
                
        last_tensor = tf.keras.layers.Dense(
                units = units,
                activation="relu", 
                name='dense_'+str(id_tmp),
            )(last_tensor)
        
        last_tensor = tf.keras.layers.BatchNormalization(
                name = 'batch_norm_dense_'+str(id_tmp)
            )(last_tensor)
            
        last_tensor = tf.keras.layers.Dropout(
                rate = 0.5,
                name = 'dropout_'+str(id_tmp)
            )(last_tensor)
        
        
    if target == 'classification':
        activation_output = 'sigmoid'
    elif target == 'regression':
        activation_output = 'linear'
        
    outputs = tf.keras.layers.Dense(
            1, 
            activation = activation_output,
            name = 'output'
        )(last_tensor)

   
    my_DNN = tf.keras.Model(
            inputs = [
                    input_x_hat,
                    input_y,
                    input_y_hat,
                ],
            outputs = outputs,
            name = name,
        )

    return my_DNN




def exp_enc_model(input_shape,
                 R,
                 depth = 5,
                 name = None,
                 cartesian_type = None,
                 fixed_mask_from_outside = False,
                 mask = None,
                ):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        R: desired acceleration rate
        pmask_slope: slope of logistic parameter in probability mask
        sample_slope: slope of logistic parameter in mask realization
        hard_threshold: whether to use binary masks (only for inference)
        
    #Returns:
        #keras model

    UNet leaky two channel
    """
    
    inputs = Input(shape=input_shape, name='input')

    last_tensor = inputs
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
        input_shape = input_shape[:-1]+(2,)

    # input -> kspace via FFT
    last_tensor_Fx = layers.FFT(name='fft')(last_tensor)
    
    if fixed_mask_from_outside == False:
        last_tensor_mask = _mask_from_tensor(last_tensor, R, cartesian_type = cartesian_type)
        
    else:
        assert mask is not None, 'provide a mask'
        last_tensor_mask = layers.FixedMask(mask)(last_tensor_Fx)

    y = layers.UnderSampleHolistic(name='undersample')([last_tensor_Fx,
                                                        last_tensor_mask])
    
    last_tensor = layers.IFFT(name='ifft')(y)
    
    outputs = [last_tensor_mask, y, last_tensor]
    
    model_enc = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    ####
    
    input_shape_mod = list(input_shape)
    input_shape_mod[-1] = 2
    input_shape_mod = tuple(input_shape_mod)
    
    inputs = Input(shape=input_shape_mod, name='input_u_net')

    last_tensor = inputs
    
    unet_tensor = _unet_from_tensor(last_tensor, depth = depth)

    last_tensor = Add(name='unet_output')([last_tensor, unet_tensor])
    
    last_tensor = layers.FFT(name='fft_projection')(last_tensor)
         
    last_tensor = layers.UnderSampleHolistic(
            complement = 1,                                 
            name='undersample_projection_complement',)([last_tensor,
                                                        last_tensor_mask])
    
    last_tensor = Add(name='add_projection')([last_tensor, y])
    
    last_tensor = layers.IFFT(name='ifft_projection')(last_tensor)
    
    last_tensor = layers.ComplexAbs(name='abs_projection')(last_tensor)
    
    if name is None:
        name = 'dec2'
    else:
        name = name
        
    return Model(inputs = inputs, outputs = last_tensor, name = name)