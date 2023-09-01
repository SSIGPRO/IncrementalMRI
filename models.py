"""
Author: Filippo Martinini

    For more details about the Incremental/Self-Assessment, please read:

        ...

    For more details about the Encoder/Decoder, please read:
    
        F. Martinini, M. Mangia, A. Marchioni, R. Rovatti and G. Setti, 
        "A Deep Learning Method for Optimal Undersampling Patterns
        and Image Recovery for MRI Exploiting Losses and Projections," 
        in IEEE Journal of Selected Topics in Signal Processing, 
        vol. 16, art. no. 4, pp. 713-724, June 2022,
        DOI: 10.1109/JSTSP.2022.3171082.
    
    Ecnoder/Decoder were inspired by LOUPE.
    For the original version of LOUPE refer to:
    
        Bahadir, C.D., Wang, A.Q., Dalca, A.V., Sabuncu, M.R. 
        "Deep-Learning-Based Optimization of the Under-Sampling Pattern in MRI"
        (2020) IEEE Transactions on Computational Imaging,
        vol. 6, art. no. 9133281, pp. 1139-1152. 
        DOI:10.1109/TCI.2020.3006727.
"""

import tensorflow as tf
import numpy as np
import layers

def data_augmentation(input_shape, factor_flip=True, factor_contrast=0.1, factor_rotation=0.1, factor_zoom=0.1):
    
    """
    A cascade of pre-processing layers.
    RandomFlip
    RandomContrast
    RandomRotation
    RandomZoom
    
    Each comes with its own factor. If the fatcor is 0, the layer is not used.
    """
    
    tensor_input = tf.keras.Input(input_shape, name='input_augmentation')
    
    tensor = tensor_input
    if factor_flip:
        tensor = tf.keras.layers.RandomFlip(name='random_flip')(tensor)
    if factor_contrast:
        tensor = tf.keras.layers.RandomContrast(factor_contrast, name='random_contrast')(tensor)
    if factor_rotation:
        tensor = tf.keras.layers.RandomRotation(factor_rotation, name='random_rotation')(tensor)
    if factor_zoom:
        tensor = tf.keras.layers.RandomZoom(factor_zoom, name='random_zoom')(tensor)
        
    return tf.keras.Model(tensor_input, tensor, name='data_augmentation')



############# LOUPE #############

def mask_mri(tensor_input, R, masker=None, pmask_slope=5, sample_slope=200, sequence_type='unconstrained'):
    
    """
    Mask used in LOUPE, with the masker for incremental
    """
    
    if masker is None:
        masker = tf.ones(tensor_input.shape)
    
    # build probability mask
    layer_prob_mask = layers.ProbMask(
        slope=pmask_slope,
        sequence_type=sequence_type,
        name='prob_mask')
    tensor_prob_mask=layer_prob_mask(tensor_input)
    
    
    # probability mask rescaled to have mean=sparsity
    rescale_layer = layers.RescaleProbMap(R, name='prob_mask_scaled',)
    tensor_prob_mask = rescale_layer(tensor_prob_mask)

    # Realization of random uniform mask
    layer_random = layers.RandomMask(
        sequence_type = sequence_type,
        name='random_mask')
    tensor_random = layer_random(tensor_prob_mask)
    
    # Realization of mask
    layer_threshold = layers.ThresholdRandomMask(slope=sample_slope, name='sampled_mask',)
    tensor_mask = layer_threshold([tensor_prob_mask, tensor_random])
    
    # Masker
    layer_masker = layers.Masker(masker=masker, name='masker')
    tensor_mask = layer_masker(tensor_mask)
    
    return tensor_mask

def enc_mri(input_shape, 
            R,
            pmask_slope=5,
            sample_slope=200,
            masker=None,
            sequence_type='unconstrained',
            name='enc'):
    
    """
    Encoder with undersampling (with mask and masker) for accelerated MRI
    """
    
    if masker is None:
        masker = np.ones(input_shape)

    tensor_masker = tf.convert_to_tensor(masker, dtype=tf.float32)

    ### layers
    
    tensor_input = tf.keras.Input(shape=input_shape, name='input_enc')

    last_tensor = tensor_input
    
    # if necessary, concatenate with zeros for FFT
    if input_shape[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
    
    # fft
    tensor_Fx = layers.FFT(name='fft')(last_tensor)
    
    # mask
    tensor_mask = mask_mri(tensor_input, R, masker, pmask_slope, sample_slope, sequence_type)

    # undersample
    layer_undersample = layers.UnderSampleHolistic(name='undersample')
    tensor_z = layer_undersample([tensor_Fx, tensor_mask])

    outputs = [tensor_mask, tensor_z]
    
    return tf.keras.Model(inputs=tensor_input, outputs=outputs, name=name)


def unet_mri(tensor,
             filt = 64,
             kern = 3,
             depth = 5,
             trainable = True,
             acti = None,
             output_nb_feats = 2,
             batch_norm_before_acti = True,
             pool_size = (2, 2),):
    
    """
    basic UNET
    """

    output_tensor = tensor
    tensor_list = []
    for i in np.arange(1, depth):
        tensor = basic_UNET_block(
                output_tensor, filt*(2**(i-1)),
                kern, acti, i, trainable, 
                batch_norm_before_acti = batch_norm_before_acti)

        output_tensor = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size, name='pool_'+str(i))(tensor)

        tensor_list += [tensor]

    output_tensor = basic_UNET_block(
            output_tensor, filt*(2**(depth-1)), 
            kern, acti, depth, trainable, 
            batch_norm_before_acti=batch_norm_before_acti)

    tensor_list = tensor_list[::-1]

    for j, i in enumerate(np.arange(depth+1, 2*depth)):

        output_tensor = tf.keras.layers.UpSampling2D(
            size=pool_size, name='up_'+str(j))(output_tensor)

        output_tensor = tf.keras.layers.Concatenate(
            axis=-1, name='concat_'+str(j))([output_tensor, tensor_list[j]])

        output_tensor = basic_UNET_block(
                output_tensor, 
                filt*(2**(depth-2-j)), kern, acti, i, trainable,
                batch_norm_before_acti=batch_norm_before_acti)

    output_tensor = tf.keras.layers.Conv2D(
        output_nb_feats, 1, padding='same', name='output_UNET', 
        trainable=trainable)(output_tensor)

    return output_tensor

def dec_mri(input_shape, name='dec'):
    
    """
    decoder for accelerated MRI, that is a residual UNET
    """
    
    tensor_input = tf.keras.Input(shape=input_shape, name='input_dec')

    last_tensor = layers.IFFT(name='ifft')(tensor_input)
        
    tensor_unet = unet_mri(last_tensor)

    last_tensor = tf.keras.layers.Add(name='add')([last_tensor, tensor_unet])
    
    return tf.keras.Model(inputs=tensor_input, outputs=last_tensor, name=name)
    
    
def dykstra_mri(shape_x, shape_mask, shape_z, name='dykstra'):
    
    """
    Dykstra alternate projection on 1) measurement constrained space; 2) real images space
    """
    
    tensor_x = tf.keras.Input(shape_x, name='input_d_x')
    tensor_mask = tf.keras.Input(shape_mask, name='input_d_mask')
    tensor_z = tf.keras.Input(shape_z, name='input_d_z')
    
    # concat zero
    last_tensor = layers.ConcatenateZero(name='d_concat_zero')(tensor_x)         

    # fft
    last_tensor = layers.FFT(name='d_fft')(last_tensor)

    # undersample
    layer_undersample = layers.UnderSampleHolistic(
        complement = True, 
        hard_threshold = True, 
        name='d_undersample',
    )
    last_tensor = layer_undersample([last_tensor, tensor_mask])

    # add
    last_tensor = tf.keras.layers.Add(name='d_add')([last_tensor, tensor_z])

    # ifft
    last_tensor = layers.IFFT(name='d_ifft')(last_tensor)

    # abs
    last_tensor = layers.ComplexAbs(name='d_abs')(last_tensor)

    # clip
    last_tensor = layers.Clip(name='d_clip')(last_tensor)
    
    return tf.keras.Model([tensor_x, tensor_mask, tensor_z], last_tensor, name=name)
    
    
def measure_constraint_mri(shape_x, shape_mask, iteratinos_dykstra=10, name='meas_const'):
    
    """
    Force the reconstructed image to respect the measurement constraint + Dykstra
    If dystra_iterations=0, no Dykstra is used.
    """
    
    tensor_x = tf.keras.Input(shape=shape_x, name='input_mc_x')
    tensor_z = tf.keras.Input(shape=shape_x, name='input_mc_z')
    tensor_mask = tf.keras.Input(shape=shape_mask, name='input_mc_mask')
    
    # fft
    last_tensor = layers.FFT(name='fft_mc')(tensor_x)
    
    # undersample
    layer_undersample = layers.UnderSampleHolistic(complement=1, name='undersample_mc')
    last_tensor = layer_undersample([last_tensor, tensor_mask])
    
    # add
    last_tensor = tf.keras.layers.Add(name='add_mc')([last_tensor, tensor_z])
    
    # ifft
    last_tensor = layers.IFFT(name='ifft_mc')(last_tensor)
    
    # abs
    last_tensor = layers.ComplexAbs(name='abs_mc')(last_tensor)
    
    # Dykstra
    model_dykstra = dykstra_mri(shape_mask, shape_mask, shape_x)
    
    for _ in range(iteratinos_dykstra):
        last_tensor = model_dykstra([last_tensor, tensor_mask, tensor_z])
        
    return tf.keras.Model([tensor_x, tensor_z, tensor_mask], last_tensor, name=name) 
    
    
    
def mae_mri(input_shape_z, input_shape_m, layer_z_mod=None, layer_x_mod=None, name='mae'):
    
    """
    "input_shape_y" is usually (x, x, 2)
    "input_shape_m" is usually (x, x, 1)
    "layer_z_mod" is a layer that takes as input the model input z, it can take the keyword "fft" or "ifft"
    "layer_x_mod" is a layer that takes as input the model input x_bar, it can take the keyword "fft" or "ifft"
    """
    
    if layer_z_mod is None:
        layer_z_mod = lambda x: x
    elif type(layer_z_mod) is str:
        
        if layer_z_mod!='fft':
            layer_z_mod = layers.FFT(name='fft_z_'+name)
            
        elif layer_z_mod!='ifft':
            layer_z_mod = layers.IFFT(name='ifft_z_'+name)
            
        else:
            raise ValueError('if "layer_z_mode" is a keyword, only "fft" or "ifft" are allowed.')
            
    if layer_x_mod is None:
        layer_x_mod = lambda x: x
    elif type(layer_x_mod) is str:
        
        if layer_x_mod!='fft':
            layer_x_mod = layers.FFT(name='fft_x_'+name)
            
        elif layer_x_mod!='ifft':
            layer_x_mod = layers.IFFT(name='ifft_x_'+name)
            
        else:
            raise ValueError('if "layer_x_mode" is a keyword, only "fft" or "ifft" are allowed.')
    
    input_z = tf.keras.Input(input_shape_z, name='input_truth_'+name)
    input_mask = tf.keras.Input(input_shape_m, name='input_mask_'+name)
    input_x_bar = tf.keras.Input(input_shape_z, name='input_recon_'+name)

    inputs = [input_z, input_mask, input_x_bar]

    input_z = layer_z_mod(input_z)
    input_x_bar = layer_x_mod(input_x_bar)
    
    # undersampling
    undersample = layers.UnderSampleHolistic(
        name='undersample_'+name,
        hard_threshold=True)
    last_tensor = undersample([input_x_bar, input_mask])
    
    # subtract
    last_tensor = tf.keras.layers.Subtract(
        name='subtract_mae_mri')([input_z, last_tensor])

    # mae on measurements
    last_tensor = layers.M_mae(name='layer_'+name)(last_tensor)

    return tf.keras.Model(inputs=inputs, outputs=last_tensor, name=name)

def mae_standard(input_shape, name='mae_standard'):
    
    input_true = tf.keras.Input(input_shape, name='input_true_'+name)
    input_pred = tf.keras.Input(input_shape, name='input_pred_'+name)

    inputs = [input_true, input_pred]
    
    # subtract
    last_tensor = tf.keras.layers.Subtract(name='subtract_'+name)(inputs)

    # mae on measurements
    last_tensor = layers.M_mae(name='layer_'+name)(last_tensor)

    return tf.keras.Model(inputs=inputs, outputs=last_tensor, name=name)

def mse_standard(input_shape, name='mse_standard'):
    
    input_true = tf.keras.Input(input_shape, name='input_true_'+name)
    input_pred = tf.keras.Input(input_shape, name='input_pred_'+name)

    inputs = [input_true, input_pred]
    
    # subtract
    last_tensor = tf.keras.layers.Subtract(name='subtract_'+name)(inputs)

    # mae on measurements
    last_tensor = layers.M_mse(name='layer_'+name)(last_tensor)

    return tf.keras.Model(inputs=inputs, outputs=last_tensor, name=name)

def ssim_standard(input_shape, name='ssim_standard'):
    
    input_true = tf.keras.Input(input_shape, name='input_true_'+name)
    input_pred = tf.keras.Input(input_shape, name='input_pred_'+name)

    inputs = [input_true, input_pred]

    # mae on measurements
    last_tensor = layers.M_ssim(name='layer_'+name)([input_true, input_pred])

    return tf.keras.Model(inputs=inputs, outputs=last_tensor, name=name)
        
    
############# U-NET #############

def basic_UNET_block(inp, filt, kern, acti, identifier,
                     trainable=True, 
                     batch_norm_before_acti=False):
            
    idf = str(identifier)

    conv = tf.keras.layers.Conv2D(
        filt, kern, activation=acti, padding='same',
        name='conv_'+idf+'_1', trainable=trainable)(inp)
    conv = batch_norm_and_relu(
        conv, idf, '1', batch_norm_before_acti=batch_norm_before_acti)
    conv = tf.keras.layers.Conv2D(
        filt, kern, activation=acti, padding='same',
        name='conv_'+idf+'_2', trainable=trainable)(conv)
    conv = batch_norm_and_relu(
        conv, idf, '2',
        batch_norm_before_acti=batch_norm_before_acti)

    return conv
        
def batch_norm_and_relu(conv, idf, idf_2, batch_norm_before_acti=False,):
    ReLu = tf.keras.layers.LeakyReLU(name = 'leaky_re_lu_'+idf+'_'+idf_2)

    if batch_norm_before_acti == False:
        conv = ReLu(conv)

    conv = tf.keras.layers.BatchNormalization(name='batch_norm_'+idf+'_'+idf_2)(conv)

    if batch_norm_before_acti == True:
        conv = ReLu(conv)

    return conv


############# Self-Assessment #############

def model_self_assessment(
    input_shape_image,
    input_shape_metric,
    line_coeff=None,
    filt=[10,8,6,4,2,1],
    kern=[5]*6,
    pool_size=[(2,2)]*6,
    average_pooling=True,
    poly_degree=2,
    dense=False,
    trainability_poly_dense=True,
    trainability_concat_dense=True,
    conv_per_block_image=1,
    name='self_assessment',
    last_activation=None):
    
    
    if line_coeff is None:
        line_coeff = [0, 1] + [0]*(poly_degree-1) 
    elif len(line_coeff)!=poly_degree+1:
        raise ValueError('poly_degree and line_coeff must have the same order.')
    
    if last_activation is None:
        last_activation = 'linear'
    
    inputs = [tf.keras.Input(shape=input_shape_image, name='input_image'), 
              tf.keras.Input(shape=input_shape_metric, name='input_metric')]
    
    last_tensor = layers.PolynomialPower(poly_degree, name='poly_power')(inputs[1])
        
    poly = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer=layers._InitializerLinRegDense(line_coeff[1:]),
        bias_initializer=tf.keras.initializers.Constant(value=line_coeff[0]),
        trainable=trainability_poly_dense, 
        name='linear_reg_dense',)
    
    last_tensor = poly(last_tensor)
    

    conv = inputs[0]
    n = len(filt)-1
    
    for ind_block, (f, k, p) in enumerate(zip(filt, kern, pool_size)):
        
        for j in range(conv_per_block_image):
            
            if conv_per_block_image==1:
                ind_conv_block = ''
            else:
                ind_conv_block = '_'+str(j)
            
            ind_layer_tmp = '_'+str(ind_block)+ind_conv_block
            
            conv = tf.keras.layers.Conv2D(
                f, k, padding='same', name='conv'+ind_layer_tmp)(conv)
            conv = tf.keras.layers.LeakyReLU(name='relu'+ind_layer_tmp)(conv)
            conv = tf.keras.layers.BatchNormalization(
                name='batch_norm'+ind_layer_tmp)(conv)
        
        if p!=(0,0) or ind_block==n:
            if average_pooling:
                if ind_block==n:
                    if dense:
                        flat = tf.keras.layers.Flatten(name='flatten')(conv)

                        last_tensor_image = tf.keras.layers.Dense(
                            1, name='dense_image_1')(flat)
                        
                    else:
                        last_tensor_image = tf.keras.layers.GlobalAveragePooling2D(
                            name='global_average_pool')(conv)
                else:
                    conv = tf.keras.layers.AveragePooling2D(
                        pool_size=p, name='average_pool_'+str(ind_block))(conv)

            else:
                if ind_block==n:
                    if dense:
                        flat = tf.keras.layers.Flatten(name='flatten')(conv)

                        last_tensor_image = tf.keras.layers.Dense(
                            1, name='dense_image_1')(flat)
                        
                    else:
                        last_tensor_image = tf.keras.layers.GlobalMaxPooling2D(
                            name='global_max_pool')(conv)

                else:
                    conv = tf.keras.layers.MaxPooling2D(
                        pool_size=p, name='max_pool_'+str(ind_block))(conv)
                
  
    concat = tf.keras.layers.Concatenate(axis=-1, name='concat')([last_tensor_image, last_tensor])

    last_tensor = tf.keras.layers.Dense(
        1,
        name='correction',
        activation=last_activation,
        kernel_initializer=tf.keras.initializers.Constant(value=[1]),
        bias_initializer=tf.keras.initializers.Constant(value=[0]),
        trainable=trainability_concat_dense)(concat)
    
    outputs = [last_tensor]
    
    if name is None:
        name = 'model_self_assessment'
    else:
        name = name
    
    return tf.keras.Model(inputs=inputs,outputs=outputs, name = name)
