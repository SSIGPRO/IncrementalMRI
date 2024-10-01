import os
import numpy as np
import tensorflow as tf
import h5py
from modules import models
from modules import utility as uty
from matplotlib import pyplot as plt
import pickle as pkl
import time
import sys
import argparse
import copy
from tqdm import tqdm


R_default = '5.5'
R_masker_default = 5.25
R_masker_of_the_masker_default = 5.0
dec_default = 2
L_default = 0
phi_default = 0
batch_size_default = 32
lr_default = 0.001
max_epochs_train_default = 10000
directory_default = 'LOUPE-decremental-IXI-sense_columns'
additional_id_string_default = 'best'

batch_size_self_assessment_default = 128
max_epochs_train_self_assessment_default = 1000
lr_self_assessment_default = 0.001

max_model_ID_default = 1

cartesian_type_default = 'sense_columns'

operation_mode_default = 'train_only_sa-load_only_model'

dataset_default = 'IXI'

GPUs_default = '2,3,4,5'

k_fold_default = False

all_versions_sa_default = '301'
gamma_sa_MAE_default = 0.001
gamma_sa_MSE_default = 0.00001

threshold_MAE_default = 42
threshold_MSE_default = 40
threshold_SSIM_default = 0.96

metric_type_list_default = 'MAE'

def train(
    R,
    R_masker,
    R_masker_of_the_masker,
    dec = 2,
    L = 0,
    phi = 0,
    batch_size = 32,
    lr = 0.001,
    max_epochs_train = 10000,
    directory = None,
    additional_id_string = None,
    batch_size_self_assessment = 32,
    max_epochs_train_self_assessment = 1000,
    lr_self_assessment = 0.001,
    max_model_ID = 4,
    cartesian_type = None,
    operation_mode = 'train_all-load_all',
    dataset = 'PD',
    GPUs = '2,3,4,5',
    all_versions_sa = 'all',
    gamma_sa_MAE = 0.001,
    gamma_sa_MSE = 0.00001,
    k_fold = False,
    threshold_MAE = 42,
    threshold_MSE = 40,
    threshold_SSIM = 0.96,
    metric_type_list = 'MAE',
):
    
    ########## MENAGE INPUT ##########
    
    
    if R == 'all':
    
        if dataset == 'IXI' and cartesian_type == 'sense_columns':

            R_list = [
                3.0, 3.25, 3.5, 3.75,
                4.0, 4.25, 4.5, 4.75,
                5.0, 5.25, 5.5, 5.75,
                6.0, 6.5,
                7.0,
                8.0,
                9.0,
                10.0,
                12.0,
                16.0,
                20.0,
                24.0,
                32.0,
                64.0,
            ]
            
            print('len(R_list) = ', len(R_list))

            R_masker_list = [0.0, ] + R_list[:-1]
            R_masker_of_the_masker_list = [0.0 ] + R_masker_list[:-1]
    
    else:
        R = float(R)
        R_list = [R]
        R_masker_list = [R_masker]
        R_masker_of_the_masker_list = [R_masker_of_the_masker]
    
    print('menage inputs')
    
    if operation_mode.find('train_all') != -1:
        flag_fit_model = True
        flag_fit_self_assessment = True
        create_path = True
        create_path_sa = True
        
    elif (operation_mode.find('train_only_sa') != -1 or
          operation_mode.find('train_only_self_assessment') != -1):
        flag_fit_model = False
        flag_fit_self_assessment = True
        create_path = False
        create_path_sa = True 
        
    elif (operation_mode.find('train_only_model') != -1 or
          operation_mode.find('train_only_masker') != -1):
        flag_fit_model = True
        flag_fit_self_assessment = False
        create_path = True
        create_path_sa = False
        
    elif (operation_mode.find('train_nothing') != -1):
        flag_fit_model = False
        flag_fit_self_assessment = False
        create_path = False
        create_path_sa = False
    
    else:
        assert False, "enter a valid operation mode"
    
    if (operation_mode.find('load_all') != -1):
        flag_load_weights_model = True
        flag_load_weights_self_assessment = True
        
    elif (operation_mode.find('load_only_sa') != -1 or
       operation_mode.find('load_only_self_assessment') != -1):
        flag_load_weights_model = False
        flag_load_weights_self_assessment = True
        
    elif (operation_mode.find('train_only_model') or
        operation_mode.find('train_only_masker') != -1):
        flag_load_weights_model = True
        flag_load_weights_self_assessment = False
        
    elif (operation_mode.find('load_nothing') != -1):
        flag_load_weights_model = False
        flag_load_weights_self_assessment = False
        
    else:
        assert False, "enter a valid operation mode"
        
    metric_type_list = [m.strip() for m in metric_type_list.split(',')]
    for m in metric_type_list:
        assert m in ['MAE', 'MSE', 'SSIM'], "every m in metric_type must be in ['MAE', 'MSE', 'SSIM']"
    
    assert (dataset == 'PD'or
            dataset == 'PDFS' or
            dataset == 'IXI'), 'enter a dataset that is one of PD, PDFS, IXI'
    
    if directory == 'None':
        directory = None
    if additional_id_string == 'None':
        additional_id_string = None
    if cartesian_type == 'None':
        cartesian_type = None
    
    directory_masker = copy.deepcopy(directory)
    
    assert ((dec==0 or dec==2) and L==0) or (dec==1 and (L==1 or L==2)), 'check for the right combinations of "dec" and "L"'
    assert R_masker < R, 'R_masker must be lower than R' 
    assert R_masker_of_the_masker <= R_masker, 'R_masker_of_the_makser must be lower or equal than R_masker'
    
    if all_versions_sa == 'all':
        all_versions_sa = [1, 301]
    else:
        all_versions_sa = [int(version_sa.strip())
                           for version_sa in all_versions_sa.split(',')
                          ]
    
    ref_id_string = [str(i) for i in range(max_model_ID+1)]
    
    assert additional_id_string in ref_id_string+['best'], 'additional_id_string NOT valid'
    
    if additional_id_string == 'best':
        
        assert flag_fit_model == False, 'if additional_id_string == "best", model must not be trained'
        
        print('get best model')
        
        # additional_id_string = str(
        #     uty.find_best_model_id(
        #         R,
        #         R_masker,
        #         max_model_ID,
        #         directory,
        #         verbose = False,
        #     )
        # )
        flag_evaluate_Dykstra = False
    else:
        flag_evaluate_Dykstra = True
    
    ########## GPU HANDLE ##########
    
    uty.handle_GPUs(GPUs = GPUs, enable_GPU=1,)
    
    ########## INPUT PRINT ##########
    
    print(
        '\nR = ', R,
        '\nR_masker = ', R_masker,
        '\nR_masker_of_the_masker = ', R_masker_of_the_masker,
        '\ndec = ', dec,
        '\nL = ', L,
        '\nphi = ', phi,
        '\nbatch_size = ', batch_size,
        '\nlr = ', lr,
        '\nmax_epochs_train = ', max_epochs_train,
        '\ndirectory = ', directory,
        '\nadditional_id_string = ', additional_id_string,
        '\nbatch_size_self_assessment = ', batch_size_self_assessment,
        '\nmax_epochs_train_self_assessment = ', max_epochs_train_self_assessment,
        '\nlr_self_assessment = ', lr_self_assessment,
        '\nmax_model_ID = ', max_model_ID,
        '\ncartesian_type = ', cartesian_type,
        '\nflag_fit_model = ', flag_fit_model,
        '\nflag_fit_self_assessment = ', flag_fit_self_assessment,
        '\nflag_load_weights_model = ', flag_load_weights_model,
        '\nflag_load_weights_self_assessment = ', flag_load_weights_self_assessment,
        '\ndataset = ', dataset,
        '\nall_versions_sa = ', all_versions_sa,
        '\nGPUs = ', GPUs,
        '\ngamma_sa_MAE = ', gamma_sa_MAE,
        '\ngamma_sa_MSE = ', gamma_sa_MSE,
        '\nmetric_type_list = ',metric_type_list,
        '\n',
    )
    
    
    ########## WAIT FOR THE START ##########
    
    data_type_list = ['train', 'val', 'test']
    
    def my_extract(key, verbose = False):
        
        if verbose == True:
            print('\nextracting ', key)
        
        store_dict = {}
        
        def my_tqdm(iterable, verbose):
            if verbose == True:
                return tqdm(iterable)
            else:
                return iterable
        
        for i, (R, R_masker, R_masker_of_the_masker) in my_tqdm(enumerate(zip(
                R_list,
                R_masker_list,
                R_masker_of_the_masker_list,
            )), verbose):

            additional_id_string = str(
                    uty.find_best_model_id(
                        R,
                        R_masker,
                        max_model_ID,
                        directory,
                        verbose = False,
                    )
                )

            path_self_assessment = os.path.join(
                    'save_models',
                    directory,
                    f'R_{R}-R_masker_{R_masker}-ID_{additional_id_string}',
                    'self-assessment',
                )

            with h5py.File(os.path.join(path_self_assessment, 'dataset_for_all_R_SA.h5'), 'r') as f:

                for data_type in data_type_list:

                    tmp = np.array(f[key][data_type][str(R)])                
                    
                    if i == 0:

                        store_dict[data_type] = np.empty((len(tmp)*len(R_list), ) + np.shape(tmp)[1:])
                        
                        pass

                    store_dict[data_type][i*len(tmp): (i+1)*len(tmp)] = tmp
        
        return store_dict
        
    verbose = False
    
    x_hat_dict = my_extract('x_hat', verbose)
    Finv_y_dict = my_extract('Finv_y', verbose)
    Finv_y_hat_dict = my_extract('Finv_y_hat', verbose)
    x_MAE_dict = my_extract('x_MAE', verbose)
    x_MSE_dict = my_extract('x_MSE', verbose)
    x_SSIM_dict = my_extract('x_SSIM', verbose)
    y_MAE_dict = my_extract('y_MAE', verbose)
    y_MSE_dict = my_extract('y_MSE', verbose)
    y_SSIM_dict = my_extract('y_SSIM', verbose)
    
    
#     shape_tmp = (len(x_MAE_dict['train']), ) + (256, 256, )
    
#     x_hat_dict = {}
#     Finv_y_dict = {}
#     Finv_y_hat_dict = {}
    
#     x_hat_dict['train'] = np.empty((shape_tmp + (1, )))
#     x_hat_dict['val'] = np.empty((shape_tmp + (1, )))
#     x_hat_dict['test'] = np.empty((shape_tmp + (1, )))
#     Finv_y_dict['train'] = np.empty((shape_tmp + (2, )))
#     Finv_y_dict['val'] = np.empty((shape_tmp + (2, )))
#     Finv_y_dict['test'] = np.empty((shape_tmp + (2, )))
#     Finv_y_hat_dict['train'] = np.empty((shape_tmp + (2, )))
#     Finv_y_hat_dict['val'] = np.empty((shape_tmp + (2, )))
#     Finv_y_hat_dict['test'] = np.empty((shape_tmp + (2, )))
    
    # END
    
    regressor_mse = uty.fit_regressor(
        y_MAE_dict['val'],
        x_MSE_dict['val'],    
    )
    line_coeff_mse = [regressor_mse.coef_[0], regressor_mse.intercept_]

    regressor_mae = uty.fit_regressor(
        y_MAE_dict['val'],
        x_MAE_dict['val'],    
    )
    line_coeff_mae = [regressor_mae.coef_[0], regressor_mae.intercept_]
    
    regressor_SSIM = uty.fit_regressor(
        y_SSIM_dict['val'],
        x_SSIM_dict['val'],    
    )
    line_coeff_SSIM = [regressor_SSIM.coef_[0], regressor_SSIM.intercept_]
    
    
    data_pred = {}
    data_true = {}
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    def my_compare(x, threshold, metric_type):
        assert metric_type in ['MAE', 'MSE', 'SSIM']
        if metric_type == 'MAE':
            if threshold > 0:
                threshold = 1/(10**(threshold/20))
            comparison = x<threshold
            
        if metric_type == 'MSE':
            if threshold > 0:
                threshold = 1/(10**(threshold/10))
            comparison = x<threshold
            
        elif metric_type == 'SSIM':
            comparison = x>threshold
        return comparison
    
    for version_sa in all_versions_sa:
        
        print('----------- VERSION S-A -----------\n\t\t',version_sa)
        
        tf.keras.backend.clear_session()
        
        def my_zip_tmp(
                metrics_for_sa = ['MAE', 'MSE', 'SSIM'],
            ):
            
            y_dict_list = []
            x_dict_list = []
            line_coeff_list = []
            threshold_list = []
            
            for m in metrics_for_sa:
                if m == 'MAE':
                    y_dict_list += [y_MAE_dict]
                    x_dict_list += [x_MAE_dict]
                    line_coeff_list += [line_coeff_mae]
                    threshold_list += [threshold_MAE]
                    
                if m == 'MSE':
                    y_dict_list += [y_MAE_dict]
                    x_dict_list += [x_MSE_dict]
                    line_coeff_list += [line_coeff_mse]
                    threshold_list += [threshold_MSE]
                    
                if m == 'SSIM':
                    y_dict_list += [y_SSIM_dict]
                    x_dict_list += [x_SSIM_dict]
                    line_coeff_list += [None]
                    threshold_list += [threshold_SSIM]
                    
            to_return = zip(
                    metrics_for_sa,
                    y_dict_list,
                    x_dict_list,
                    line_coeff_list,
                    threshold_list,
                )
            
            return to_return
        
        for (metric_type, 
             y_dict,
             x_dict,
             line_coeff,
             threshold,
            ) in my_zip_tmp(metric_type_list):
        
            print('------ METRIC TYPE -----\n\t',metric_type)
                
            ########## DATASET FOR SA #########

            print('create dataset for sa')

            if version_sa in [1, 200]:
                
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment,] = [
                        ([x_hat, y], x) 
                    for x_hat, y, x in zip(
                            [x_hat_dict['test'], x_hat_dict['val'], x_hat_dict['train']],
                            [y_dict['test'], y_dict['val'], y_dict['train']],
                            [x_dict['test'], x_dict['val'], x_dict['train']],     
                        )]
                if k_fold == True:
                    xtdata_self_assessment = tdata_self_assessment
                    
            if version_sa in [300]:
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment,] = [
                        ([x_hat, y], my_compare(x, threshold, metric_type)) 
                    for x_hat, y, x in zip(
                            [x_hat_dict['test'], x_hat_dict['val'], x_hat_dict['train']],
                            [y_dict['test'], y_dict['val'], y_dict['train']],
                            [x_dict['test'], x_dict['val'], x_dict['train']],     
                        )]
                
                if k_fold == True:
                    xtdata_self_assessment = tdata_self_assessment
                    
            elif version_sa in [301]: # model used for I2TMC and binary
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment,] = [
                        ([x_hat, Finv_y, Finv_y_hat], my_compare(x, threshold, metric_type)) 
                    for x_hat, Finv_y, Finv_y_hat, x in zip(
                            [x_hat_dict['test'], x_hat_dict['val'], x_hat_dict['train']],
                            [Finv_y_dict['test'], Finv_y_dict['val'], Finv_y_dict['train']],
                            [Finv_y_hat_dict['test'], Finv_y_hat_dict['val'], Finv_y_hat_dict['train']], 
                            [x_dict['test'], x_dict['val'], x_dict['train']],
                        )]
                
                print('\n\npercentuage of 1s in the train, val and test set:')
                perc_ones_train = np.sum(xdata_self_assessment[1])/len(xdata_self_assessment[1])*100
                perc_ones_val = np.sum(vdata_self_assessment[1])/len(vdata_self_assessment[1])*100
                perc_ones_test = np.sum(tdata_self_assessment[1])/len(tdata_self_assessment[1])*100
                
                if (perc_ones_train<25 or perc_ones_train>75) or (perc_ones_val<20 or perc_ones_val>80):
                    
                    flag_fit_self_assessment = False
                    print('DATASET IS TOO UNBALANCED: TRAINING WOULD NOT LEAD TO ANYTHING GOOD!')
                    print(f'perc_ones_train = {perc_ones_train}\nperc_ones_val = {perc_ones_val}\nperc_ones_test = {perc_ones_test}')
                else:
                    print('these are good proportions!')
                    print(f'perc_ones_train = {perc_ones_train}\nperc_ones_val = {perc_ones_val}\n perc_ones_test = {perc_ones_test}')
                
                if k_fold == True:
                    xtdata_self_assessment = tdata_self_assessment

            ########## MODELS ##########
            
            if k_fold == True:
            
                def select_k_fold(d, ind):
                    d_1 = [d_tmp[ind] for d_tmp in d[0]]
                    d_2 = d[1][ind]
                    return (d_1, d_2)
                
                if dataset == 'IXI':
                    vol_start_ind = np.arange(0, 9)*150

                elif dataset == 'PD':
                    vol_start_ind = [0, 36, 71, 106, 151, len(xtdata_self_assessment[1])]

                # BEGIN - volumes indexes for k fold
                vol_ind = []
                for s in range(len(vol_start_ind)-1):
                    test_index = np.arange(vol_start_ind[s], vol_start_ind[s+1])
                    vol_ind += [test_index]

                k_fold_ind_list = []
                for i in range(len(vol_start_ind)-1):
                    test_index = vol_ind[i]
                    train_index = np.delete(np.arange(0, vol_start_ind[-1]), test_index)
                    k_fold_ind_list += [[train_index, test_index]]
                # END - volumes indexes for k fold  
            else:
                
                k_fold_ind_list = [[None, None]]
                
            for k_fold_iter, (train_index, test_index) in enumerate(k_fold_ind_list):
                
                if k_fold == True:
                    
                    xdata_self_assessment = select_k_fold(xtdata_self_assessment, train_index)
                    tdata_self_assessment = select_k_fold(xtdata_self_assessment, test_index)

                    print(f'\n    -----    FOLD {k_fold_iter}    -----\n')
                else:
                    k_fold_iter = None
                print(directory, R)
                print('retrieve models self-assessment')
                
                with mirrored_strategy.scope():

                    if version_sa in [1, 200, 300]:

                        if version_sa in [1, 200]:
                            filt = [10,8,6,4,2,1]
                            kern = [5]*len(filt)
                            pool_size = [(2,2)]*len(filt)
                            average_pooling = True
                            poly_degree = 2
                            dense = False
                            conv_per_block_image = 1
                            units = [100, 20]
                            last_activation = None
                            
                        if version_sa in [300]:
                            filt = [10,8,6,4,2,1]
                            kern = [5]*len(filt)
                            pool_size = [(2,2)]*len(filt)
                            average_pooling = True
                            poly_degree = 2
                            dense = False
                            conv_per_block_image = 1
                            units = [100, 20]
                            last_activation = 'softmax'

                        model_self_assessment = models.model_self_assessment(
                                input_shape_image = np.shape(xdata_self_assessment[0][0])[1:],
                                input_shape_metric = (1,),
                                line_coeff = line_coeff,
                                filt = filt,
                                kern = kern,
                                pool_size = pool_size,
                                average_pooling = average_pooling,
                                poly_degree = poly_degree,
                                dense = dense,
                                conv_per_block_image = conv_per_block_image,
                                name = 'model_self_assessment_version'+str(version_sa),
                                last_activation = last_activation,
                            )

                    if version_sa in [301]:
                        
                        if version_sa in [301]:
                            filt = [10,8,6,4,2,1]
                            kern = [5]*len(filt)
                            pool_size = [(2,2)]*len(filt)
                            average_pooling = True
                            poly_degree = 2
                            dense = False
                            conv_per_block_image = 1
                            units = [100, 20]
                            last_activation = 'sigmoid'

                        model_self_assessment = models.model_self_assessment_I2TMC(
                                    input_shape_x_hat = np.shape(xdata_self_assessment[0][0])[1:],
                                    input_shape_y = np.shape(xdata_self_assessment[0][1])[1:],
                                    input_shape_y_hat = np.shape(xdata_self_assessment[0][2])[1:],
                                    target = 'classification',
                                    name = 'model_self_assessment_version'+str(version_sa),
                                )
                        
                        # model_self_assessment.summary()

                    ########## COMPILE ##########

                    def mae_with_regularization(y_true, y_pred, gamma = 0.001):

                        diff = y_true-y_pred

                        mae = tf.math.reduce_mean(tf.math.abs(diff))

                        mae = (1-gamma) * mae + gamma * tf.math.reduce_mean(tf.cast(
                            tf.math.greater(y_true, y_pred), tf.float32))

                        return mae


                    def mse_with_regularization(y_true, y_pred, gamma = 0.001):

                        diff = y_true-y_pred

                        mse = tf.math.reduce_mean(tf.math.abs(diff))

                        mse = (1-gamma) * mse + gamma * tf.math.reduce_mean(tf.cast(
                            tf.math.greater(y_true, y_pred), tf.float32))

                        return mse

                    # CHANGE SA LOSS

                    if version_sa in [1, 200]:
                        loss = 'mae'
                        metrics = []
                        threshold = None
                        
                    elif version_sa in [300, 301]:
                        loss = tf.keras.losses.BinaryCrossentropy()
                        metrics = ['accuracy']
                        
                    print('compile self-assessment')

                    model_self_assessment.compile(
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_self_assessment),
                        loss = loss,
                        metrics = metrics,
                    )

                
                ########## PATH ##########

                # add str for self assessment
                if version_sa in [-1]:
                    if metric_type == 'MAE':
                        gamma_sa_tmp = gamma_sa_MAE
                    elif metric_type == 'MSE':
                        gamma_sa_tmp = gamma_sa_MSE
                else:
                    gamma_sa_tmp = None

                additional_id_string = str(
                        uty.find_best_model_id(
                            R,
                            R_masker,
                            max_model_ID,
                            directory,
                            verbose = False,
                        )
                    )

                path_self_assessment = os.path.join(
                        'save_models',
                        directory,
                        f'R_{R}-R_masker_{R_masker}-ID_{additional_id_string}',
                        'self-assessment',
                    )
                    
                print('path self-assessment')
                paths, existance_path_tmp = uty.from_self_assessment_to_useful_paths(
                    path_self_assessment, 
                    create_path = create_path_sa,
                    version_sa = version_sa,
                    gamma_sa = gamma_sa_tmp,
                    k_fold_ind = k_fold_iter,
                    threshold = threshold,
                    metric_type_list = [metric_type],
                )

                path_self_assessment_checkpoint = paths[1]
                path_self_assessment_board = paths[2]
                
                print('path checkpoint =', path_self_assessment_checkpoint)
                print('path board =', path_self_assessment_board)
                
                # if metric_type == 'MAE':
                #     path_self_assessment_checkpoint = paths[1]
                #     path_self_assessment_board = paths[2]
                # elif metric_type == 'MSE':
                #     path_self_assessment_checkpoint = paths[4]
                #     path_self_assessment_board = paths[5]
                # elif metric_type == 'SSIM':
                #     path_self_assessment_checkpoint = paths[7]
                #     path_self_assessment_board = paths[8]

                ########## LOAD WEIGHTS ##########
                
                if type(existance_path_tmp) == list:
                    existance_path_tmp_flag = existance_path_tmp[1]
                
                if flag_load_weights_self_assessment == True and existance_path_tmp_flag == True:

                    ref_file_chpt = [
                        'checkpoint.ckpt',
                        'checkpoint.ckpt.index',
                        'checkpoint.ckpt.data-00000-of-00001',
                        'checkpoint',
                    ]

                    if os.listdir(
                        os.path.dirname(
                            path_self_assessment_checkpoint)) == ref_file_chpt:

                        print('LOAD weights self-assessment')
                        
                        model_self_assessment.load_weights(
                            path_self_assessment_checkpoint)
                    else:
                        print('self-assessment never trained before')

                else:
                    print('not loading the weights of self-assessment')
                    
                ########## CALLBACKS ##########

                print('preparing callbacks self-assessment')

                callbacks_self_assessment = uty.callbacks(
                    checkpoint = path_self_assessment_checkpoint,
                    ldir = path_self_assessment_board,
                    patienceEarlyStop = 65,
                    patienceLR = 40,
                )


                ########## FIT ##########
                if flag_fit_self_assessment == True:

                    print('training self-assessment')

                    verbose_training = 2

                    history = model_self_assessment.fit(
                        x = xdata_self_assessment[0],
                        y = xdata_self_assessment[1],
                        initial_epoch = 0,
                        epochs = max_epochs_train_self_assessment,
                        validation_data = vdata_self_assessment,
                        batch_size = batch_size_self_assessment,
                        callbacks = callbacks_self_assessment,
                        verbose = verbose_training,
                    )

                else:
                    print('mode operation skips training self-assessment')

                ########## RESULTS ##########

                print('results self-assessment')
                
                list_tmp = [
                    (data_type_tmp, data_self_assessment_tmp)
                    for data_type_tmp, data_self_assessment_tmp 
                    in zip(['test', 'val', 'train'],
                           [tdata_self_assessment, vdata_self_assessment, xdata_self_assessment],
                          )
                    if data_type_tmp in data_type_list
                ]

                for (data_type_tmp, data_self_assessment_tmp) in list_tmp:

                    id_result = metric_type+'-'+data_type_tmp+'-'+str(version_sa)
                    if k_fold == True:
                        id_result = id_result+'-k'+str(k_fold_iter)

                    data_pred[id_result] = model_self_assessment.predict(
                            data_self_assessment_tmp[0],
                            batch_size = 16
                        )
                    
                    print('\n\n  - storing result: ',id_result)

                    if version_sa in [301]:
                        
                        print('\tACCURACY =', np.sum(
                            data_self_assessment_tmp[1] == (np.array(data_pred[id_result][..., 0]) > 0.5)
                        )/len(data_self_assessment_tmp[1])*100)
                        
                    
                    data_true[id_result] = copy.deepcopy(data_self_assessment_tmp[1])

                pass
            pass
    
    ########## SAVE SELF-ASSESSMENT RESULTS ##########
    
    print('\nsave the self-assessment results\n')
    
    path_self_assessment_h5 = os.path.join(path_self_assessment, 'dataset.h5')
    print(path_self_assessment_h5)
    
    with h5py.File(path_self_assessment_h5, "a") as f:
                
        ###### SCORE ######

        print('save score')
        
        for version_sa in all_versions_sa:
            
            for metric_type in metric_type_list:
            # for metric_type, gamma_sa_tmp in zip(['MSE'], [gamma_sa_MSE]):
                
                if version_sa == 1:
                    additional_string_tmp = ''
                else:
                    additional_string_tmp = '_v'+str(version_sa)

                if version_sa in [-1]:
                    if gamma_sa_tmp != 2:
                        additional_string_tmp = additional_string_tmp + '_g' + str(gamma_sa_tmp)
                
                
                group_str = metric_type+additional_string_tmp
                if metric_type+additional_string_tmp in f.keys():
                    d_tmp = f[group_str]
                else:
                    d_tmp = f.create_group(group_str)
                    
                
                for score_type_tmp, data_tmp in zip(
                        ['predictions_score', 'true_score'],
                        [data_pred, data_true],
                    ):
                    
                    if score_type_tmp in d_tmp.keys():
                        id_tmp_score = d_tmp[score_type_tmp]
                    else:
                        id_tmp_score = d_tmp.create_group(score_type_tmp)
                    
                    for data_type in data_type_list:
                
                        id_result = metric_type+'-'+data_type+'-'+str(version_sa)
                        
                        if k_fold == False:
                            k_fold_ind_list = [None]
                        
                        for k_fold_iter in range(len(k_fold_ind_list)):
                            
                            id_result_tmp = id_result
                            if k_fold == True:
                                id_result_tmp = id_result_tmp+'-k'+str(k_fold_iter)
                    
                    
                            dataset_name_tmp = data_type+'-k_fold_'+str(k_fold_iter)
                            print(group_str,score_type_tmp,id_result_tmp,'- shape: ', np.shape(data_tmp[id_result_tmp]))
                            
                            if not(dataset_name_tmp in id_tmp_score.keys()):
                                id_tmp_score.create_dataset(dataset_name_tmp, data = data_tmp[id_result_tmp])
                            else:
                                del id_tmp_score[dataset_name_tmp]
                                id_tmp_score.create_dataset(dataset_name_tmp, data = data_tmp[id_result_tmp])
            
                        
            
            pass
        pass
        
    
    print('END')
    pass
    



def main(args):
    
    version = '0.1'
        
    parser = argparse.ArgumentParser(description = '')

    parser.add_argument(
        '--version', action='version', version='%(prog)s '+ version,
        help='program version')
    
# R
    
    parser.add_argument(
        '-R', 
        dest='R', 
        action='store', 
        type=str, 
        default=R_default,
        help='speed-up',
    )

# R_masker

    parser.add_argument(
        '--R_masker',
        dest='R_masker', 
        action='store', 
        type=float, 
        default=R_masker_default,
        help='speed-up',
    )

# R_masker_of_the_masker

    parser.add_argument(
        '--R_masker_of_the_masker', 
        dest='R_masker_of_the_masker', 
        action='store',
        type=float, 
        default=R_masker_of_the_masker_default,
        help='speed-up masker of the masker',
    )

    
# dec
    parser.add_argument(
        '--dec', 
        dest='dec', 
        action='store',
        type=int, 
        default=dec_default,
        help='dec type (0, 1, 2)',
    )

# L
    parser.add_argument(
        '-L', 
        dest='L', 
        action='store',
        type=int, 
        default=L_default,
        help='Loss type (0, 1, 2)',
    )

# phi

    parser.add_argument(
        '--phi', 
        dest='phi', 
        action='store',
        type=float, 
        default=phi_default,
        help='regularization weight',
    )

# batch_size

    parser.add_argument(
        '--batch_size', 
        dest='batch_size', 
        action='store',
        type=int, 
        default=batch_size_default,
        help='batch size',
    )

# lr

    parser.add_argument(
        '--lr', 
        dest='lr', 
        action='store',
        type=float, 
        default=lr_default,
        help='learning rate',
    )

# max_epochs_train

    parser.add_argument(
        '--max_epochs_train', 
        dest='max_epochs_train', 
        action='store',
        type=int, 
        default=max_epochs_train_default,
        help='max_epochs_train',
    )

# directory

    parser.add_argument(
        '--directory', 
        dest='directory', 
        action='store',
        type=str, 
        default=directory_default,
        help='directory',
    )

# additional_id_string

    parser.add_argument(
        '--id_string', 
        dest='additional_id_string', 
        action='store',
        type=str, 
        default=additional_id_string_default,
        help='additional_id_string',
    )
    
# batch_size_self_assessment

    parser.add_argument(
        '--batch_size_self_assessment', 
        dest='batch_size_self_assessment', 
        action='store',
        type=int, 
        default=batch_size_self_assessment_default,
        help='batch_size_self_assessment',
    )
    
# max_epochs_train_self_assessment

    parser.add_argument(
        '--max_epochs_train_self_assessment', 
        dest='max_epochs_train_self_assessment', 
        action='store',
        type=int, 
        default=max_epochs_train_self_assessment_default,
        help='max_epochs_train_self_assessment',
    )

# lr_self_assessment
    
    parser.add_argument(
        '--lr_self_assessment', 
        dest='lr_self_assessment', 
        action='store',
        type=float, 
        default=lr_self_assessment_default,
        help='lr_self_assessment',
    )
    
# max_model_ID

    parser.add_argument(
        '--max_model_ID', 
        dest='max_model_ID', 
        action='store',
        type=int, 
        default=max_model_ID_default,
        help='max_model_ID, model IDs start from 0 and evolve with +1',
    )
    
# cartesian_type

    parser.add_argument(
        '--cartesian_type', 
        dest='cartesian_type', 
        action='store',
        type=str, 
        default=cartesian_type_default,
        help='cartesian_type',
    )
    
# operation_mode

    parser.add_argument(
        '--operation_mode',
        dest='operation_mode', 
        action='store',
        type=str, 
        default=operation_mode_default,
        help='operation_mode',
    )
    
# dataset

    parser.add_argument(
        '--dataset',
        dest='dataset', 
        action='store',
        type=str, 
        default=dataset_default,
        help='dataset keyword reference',
    )

# GPUs

    parser.add_argument(
        '--GPUs', 
        dest='GPUs', 
        action='store',
        type=str, 
        default=GPUs_default,
        help='number of the GPU to use',
    )
    
# all_versions_sa

    parser.add_argument(
        '--all_versions_sa', 
        dest='all_versions_sa', 
        action='store',
        type=str, 
        default=all_versions_sa_default,
        help='all the versions of self-assessment one wants to use',
    )
    
# gamma_sa_MAE

    parser.add_argument(
        '--gamma_sa_MAE', 
        dest='gamma_sa_MAE', 
        action='store',
        type=float, 
        default=gamma_sa_MAE_default,
        help='reg_term weight for self-assessment',
    )
    
# gamma_sa_MSE

    parser.add_argument(
        '--gamma_sa_MSE', 
        dest='gamma_sa_MSE', 
        action='store',
        type=float, 
        default=gamma_sa_MSE_default,
        help='reg_term weight for self-assessment',
    )
    
# k_fold

    parser.add_argument(
        '--k_fold', 
        dest='k_fold', 
        action='store',
        type=int, 
        default=k_fold_default,
        help='whther to use k_fold or not',
    )
    
# threshold_MAE

    parser.add_argument(
        '--threshold_MAE', 
        dest='threshold_MAE', 
        action='store',
        type=float, 
        default=threshold_MAE_default,
        help='threshold for binary crossentropy (used for self_assessment version in [300]',
    )
    
# threshold_MSE

    parser.add_argument(
        '--threshold_MSE', 
        dest='threshold_MSE', 
        action='store',
        type=float, 
        default=threshold_MSE_default,
        help='threshold for binary crossentropy (used for self_assessment version in [300]',
    )
    
# threshold_SSIM

    parser.add_argument(
        '--threshold_SSIM', 
        dest='threshold_SSIM', 
        action='store',
        type=float, 
        default=threshold_SSIM_default,
        help='threshold for binary crossentropy (used for self_assessment version in [300]',
    )
    
# metric_type_list

    parser.add_argument(
        '--metric_type_list', 
        dest='metric_type_list', 
        action='store',
        type=str, 
        default=metric_type_list_default,
        help='threshold for binary crossentropy (used for self_assessment version in [300]',
    )
    
    
    
               
    args = parser.parse_args(args)    
    print(args)
        
    train(
        R = args.R,
        R_masker = args.R_masker,
        R_masker_of_the_masker = args.R_masker_of_the_masker,
        dec = args.dec,
        L = args.L,
        phi = args.phi,
        batch_size = args.batch_size,
        lr = args.lr,
        max_epochs_train = args.max_epochs_train,
        directory = args.directory,
        additional_id_string = args.additional_id_string,
        batch_size_self_assessment = args.batch_size_self_assessment,
        max_epochs_train_self_assessment = args.max_epochs_train_self_assessment,
        lr_self_assessment = args.lr_self_assessment,
        max_model_ID = args.max_model_ID,
        cartesian_type = args.cartesian_type,
        operation_mode = args.operation_mode,
        dataset = args.dataset,
        GPUs = args.GPUs,
        all_versions_sa = args.all_versions_sa,
        gamma_sa_MAE = args.gamma_sa_MAE,
        gamma_sa_MSE = args.gamma_sa_MSE,
        k_fold = args.k_fold,
        threshold_MAE = args.threshold_MAE,
        threshold_MSE = args.threshold_MSE,
        threshold_SSIM = args.threshold_SSIM,
        metric_type_list = args.metric_type_list,
    )

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))