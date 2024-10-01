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


R_default = 6.0
R_masker_default = 3.0
R_masker_of_the_masker_default = 0.0
dec_default = 2
L_default = 0
phi_default = 0
batch_size_default = 128
lr_default = 0.001
max_epochs_train_default = 10000
directory_default = 'None'
additional_id_string_default = 'None'

batch_size_self_assessment_default = 128
max_epochs_train_self_assessment_default = 1000
lr_self_assessment_default = 0.001

max_model_ID_default = 5

cartesian_type_default = None

operation_mode_default = 'train_all-load_all'

dataset_default = 'PD'

GPUs_default = '0,1,2,3,4,5'

all_versions_sa_default = 'all'

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
    GPUs = '0,1,2,3,4,5',
    all_versions_sa = 'all',
):
    
    ########## MENAGE INPUT ##########
    
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
        all_versions_sa = [1, 2, 3, 4]
    else:
        all_versions_sa = [int(version_sa.strip())
                           for version_sa in all_versions_sa.split(',')
                          ]
    
    ref_id_string = [str(i) for i in range(max_model_ID+1)]
    
    assert additional_id_string in ref_id_string+['best'], 'additional_id_string NOT valid'
    
    if additional_id_string == 'best':
        
        assert flag_fit_model == False, 'if additional_id_string == "best", model must not be trained'
        
        print('get best model')
        
        additional_id_string = str(
            uty.find_best_model_id(
                R,
                R_masker,
                max_model_ID,
                directory,
                verbose = False,
            )
        )
    
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
        '\n',
    )
    
    ########## GPU HANDLE ##########
    
    uty.handle_GPUs(GPUs = GPUs, enable_GPU=1,)
    
    ########## WAIT FOR THE START ##########
    
    print('wait for the start')
    
    if R_masker != 0.0:
        
        while True:

            ### check if all the masker models have finished training ###
            
            counter_maskers_finished_training = 0
            
            for additional_id_string_masker_tmp in range(max_model_ID+1):
                
                kwargs_model_masker_tmp = {
                    'R': R_masker,
                    'R_masker': R_masker_of_the_masker,
                }

                if directory_masker is not None:
                    kwargs_model_masker_tmp['directory'] = directory_masker
                if additional_id_string_masker_tmp is not None:
                    kwargs_model_masker_tmp['ID'] = additional_id_string_masker_tmp
                
                (path_model_tmp, 
                 flag_tmp) = uty.from_id_to_model_path(**kwargs_model_masker_tmp)
                
                if os.path.isfile(os.path.join(path_model_tmp,
                                               'model_evaluate.pkl')):
                    counter_maskers_finished_training += 1
            
            
            if counter_maskers_finished_training < (max_model_ID+1):
                print('waiting for ',(max_model_ID+1)-counter_maskers_finished_training,
                  ' maskers to finish the training..........SLEEP..........')
                time.sleep(60*5) # wait 1 minute and check again

            else:
                
                ### FIND THE BEST MASKER ###
                
                PSNR_masker = 0
                additional_id_string_masker = None
                
                for additional_id_string_masker_tmp in range(max_model_ID+1):
                
                    kwargs_model_masker_tmp = {
                        'R': R_masker,
                        'R_masker': R_masker_of_the_masker,
                    }

                    if directory_masker is not(None):
                        kwargs_model_masker_tmp['directory'] = directory_masker
                    if additional_id_string_masker_tmp is not None:
                        kwargs_model_masker_tmp['ID'] = additional_id_string_masker_tmp

                    path_model_tmp, _ = uty.from_id_to_model_path(**kwargs_model_masker_tmp)
                    
                    with open(os.path.join(path_model_tmp, 'model_evaluate.pkl'), 'rb') as f:
                        PSNR_masker_tmp = pkl.load(f)
                        
                    if PSNR_masker_tmp > PSNR_masker:
                        PSNR_masker = PSNR_masker_tmp
                        additional_id_string_masker = additional_id_string_masker_tmp
                        
                print('\nBest masker ID: ',additional_id_string_masker,
                      '\nBest masker PSNR = ', PSNR_masker,'\n')
                
                break
    
    ########## PATH ##########
    
    print('path handler')
    
    kwargs_model = {
        'R': R,
        'R_masker': R_masker,
    }
    
    if directory is not None:
        kwargs_model['directory'] = directory
    if additional_id_string is not None:
        kwargs_model['ID'] = additional_id_string

    
    path_model, flag = uty.from_id_to_model_path(create_path, 
                                                 **kwargs_model)

    assert flag == True, 'specified model does not exist'

    ([path_checkpoint, 
      path_board,
      path_mask,
      path_self_assessment], 
      flag_load_weights)= uty.from_model_path_to_useful_paths(path_model, 
                                                              create_path)

    assert flag == True, 'specified paths do not exist'
    
    ########## LOAD DATASET S-A ##########
    
    print('load dataset sa')
    
    d_name_list = [
        'tx_hat', 'Finv_ty_diff_MAE',
        'ty_MAE', 'tx_MAE',
        'Finv_ty_diff_MSE', 'ty_MSE', 
        'tx_MSE', 'vx_hat', 'Finv_vy_diff_MAE',
        'vy_MAE', 'vx_MAE',
        'Finv_vy_diff_MSE', 'vy_MSE', 
        'vx_MSE', 'xx_hat', 'Finv_xy_diff_MAE',
        'xy_MAE', 'xx_MAE', 'Finv_xy_diff_MSE',
        'xy_MSE', 'xx_MSE',
    ]
    
    path_self_assessment_h5 = os.path.join(path_model, 'self-assessment', 'dataset_to_train_sa.h5')
    
    with h5py.File(path_self_assessment_h5, "r") as f:
        
        d = f['dataset']
        
        [tx_hat, Finv_ty_diff_MAE,
         ty_MAE, tx_MAE,
         Finv_ty_diff_MSE, ty_MSE, 
         tx_MSE, vx_hat, Finv_vy_diff_MAE,
         vy_MAE, vx_MAE, 
         Finv_vy_diff_MSE, vy_MSE,
         vx_MSE, xx_hat, Finv_xy_diff_MAE,
         xy_MAE, xx_MAE, Finv_xy_diff_MSE,
         xy_MSE, xx_MSE,
        ] = [np.array(d[d_name]) for d_name in tqdm(d_name_list)]
          
        pass
    
    ############################## SELF-ASSESSMENT ##############################

    ########## LINEAR SELF-ASSESSMENT ##########

    print('linear self-assessment')

    # train linear regressor
    # line coeff are meant to be a first guess for the neural self-assessment
    # (neural self-assessment will modify them during training)


    regressor_mse = uty.fit_regressor(
        xy_MSE,
        xx_MSE,    
    )
    line_coeff_mse = [regressor_mse.coef_[0], regressor_mse.intercept_]

    regressor_mae = uty.fit_regressor(
        xy_MAE,
        xx_MAE,    
    )
    line_coeff_mae = [regressor_mae.coef_[0], regressor_mae.intercept_]
    
    data_type_list = [
        'test',
        'val',
        # 'train',
    ]
    
    data_pred = {}
    data_true = {}
    
    for version_sa in all_versions_sa:
        
        print('----------- VERSION S-A -----------\n\t\t',version_sa)
        
        # tf.keras.backend.clear_session()
        
        for (metric_type, 
             ty, vy, xy, 
             tx, vx, xx, 
             Finv_ty_diff, Finv_vy_diff, Finv_xy_diff, 
             line_coeff) in zip(
                    ['MAE', 'MSE'],
                    [ty_MAE, ty_MSE], [vy_MAE, vy_MSE], [xy_MAE, xy_MSE],
                    [tx_MAE, tx_MSE], [vx_MAE, vx_MSE], [xx_MAE, xx_MSE],
                    [Finv_ty_diff_MAE, Finv_ty_diff_MSE],
                    [Finv_vy_diff_MAE, Finv_vy_diff_MSE],
                    [Finv_xy_diff_MAE, Finv_xy_diff_MSE],
                    [line_coeff_mae, line_coeff_mse],
                ):
        
            print('------ METRIC TYPE -----\n\t',metric_type)
                
            ########## DATASET FOR SA #########

            print('create dataset for sa')

            if version_sa == 1 or version_sa == 3:
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment,] = [
                        ([x_hat, y], x) 
                    for x_hat, y, x in zip(
                            [tx_hat, vx_hat, xx_hat],
                            [ty, vy, xy],
                            [tx, vx, xx],     
                        )]

            elif version_sa == 2 or version_sa == 4:

                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment] = [
                        ([np.concatenate((x_hat, Finv_y_diff[..., 0][..., np.newaxis]), 2),
                          y],
                         x
                        )
                    for x_hat, Finv_y_diff, y, x in zip(
                            [tx_hat, vx_hat, xx_hat],
                            [Finv_ty_diff, Finv_vy_diff, Finv_xy_diff],
                            [ty, vy, xy],
                            [tx, vx, xx],     
                        )
                ]
                
                pass

            ########## MODELS ##########

            print('retrieve models self-assessment')
            
            mirrored_strategy = tf.distribute.MirroredStrategy()
            
            with mirrored_strategy.scope():

                if version_sa == 2 or version_sa == 4:
                    filt = [40,20,10,8,6,4,2,1]
                    kern = [5]*len(filt)
                    pool_size = [(2,2)]*len(filt)
                    average_pooling = False
                    poly_degree = 1

                elif version_sa == 1 or version_sa == 3:
                    filt = [10,8,6,4,2,1]
                    kern = [5]*len(filt)
                    pool_size = [(2,2)]*len(filt)
                    average_pooling = True
                    poly_degree = 2
                    
                # tf.keras.backend.clear_session()

                model_self_assessment = models.model_self_assessment(
                    input_shape_image = np.shape(tdata_self_assessment[0][0])[1:],
                    input_shape_metric = (1,),
                    line_coeff = line_coeff,
                    filt = filt,
                    kern = kern,
                    pool_size = pool_size,
                    average_pooling = average_pooling,
                    poly_degree = poly_degree,
                    name = 'model_self_assessment_version'+str(version_sa),
                )

                ########## COMPILE ##########

                def mae_with_penalty(y_true, y_pred, gamma = 2):

                    diff = y_true-y_pred

                    mae = tf.math.reduce_mean(tf.math.abs(diff))
                    
                    # if mae_pred < mae_true : the model thinks the recostruction is good, but it is not.
                    mae = mae * (
                            1 + gamma * 
                            tf.reduce_mean(tf.cast(tf.math.greater(y_true, y_pred), tf.float32))
                        )

                    return mae

                def mse_with_penalty(y_true, y_pred, gamma = 2):

                    diff = y_true-y_pred

                    mse = tf.reduce_mean(tf.math.pow(diff, 2))

                    # if mse_pred < mse_true : the model thinks the recostruction is good, but it is not.
                    mse = mse * (
                            1 + gamma * 
                            tf.reduce_mean(tf.cast(tf.math.greater(y_true, y_pred), tf.float32))
                        )
                    
                    return mse

                def diff_PSNR(y_true, y_pred):
                    
                
                # CHANGE SA LOSS

                if version_sa == 3 or version_sa == 4:
                    
                    if metric_type == 'MAE':
                        loss = lambda y_true, y_pred: mae_with_penalty(y_true, y_pred, gamma = 2)
                    elif metric_type == 'MSE':
                        loss = lambda y_true, y_pred: mse_with_penalty(y_true, y_pred, gamma = 2)
                else:

                    if metric_type == 'MAE':
                        loss = 'mae'
                    elif metric_type == 'MSE':
                        loss = 'mse'
                
                print('compile self-assessment')
                
                model_self_assessment.compile(
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_self_assessment),
                    loss = loss,
                )

            ########## PATH ##########

            print('path self-assessment')
            paths, _ = uty.from_self_assessment_to_useful_paths(
                path_self_assessment, 
                create_path = create_path_sa,
                version = version_sa,
            )
            if metric_type == 'MAE':
                path_self_assessment_checkpoint = paths[1]
                path_self_assessment_board = paths[2]
            elif metric_type == 'MSE':
                path_self_assessment_checkpoint = paths[4]
                path_self_assessment_board = paths[5]

            ########## LOAD WEIGHTS ##########

            if flag_load_weights_self_assessment == True:

                ref_file_chpt = [
                    'checkpoint.ckpt',
                    'checkpoint.ckpt.index',
                    'checkpoint.ckpt.data-00000-of-00001',
                    'checkpoint',
                ]

                if os.listdir(
                    os.path.dirname(
                        path_self_assessment_checkpoint)) == ref_file_chpt:

                    print('load weights self-assessment')
                    model_self_assessment.load_weights(
                        path_self_assessment_checkpoint)
                else:
                    print('no weights to load')
            else:
                print('load weight flag is False')

            ########## CALLBACKS ##########

            print('preparing callbacks self-assessment')

            callbacks_self_assessment = uty.callbacks(
                checkpoint = path_self_assessment_checkpoint,
                ldir = path_self_assessment_board,
                patienceEarlyStop = 50,
                patienceLR = 30,   
            )

            ########## FIT ##########
            if flag_fit_self_assessment == True:
                
                print('training self-assessment')
                
                verbose_training = False
                
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
                
                data_pred[id_result] = model_self_assessment.predict(
                        data_self_assessment_tmp[0],
                        batch_size = 16
                    )
                
                print(id_result)
                
                data_true[id_result] = copy.deepcopy(data_self_assessment_tmp[1])

                
                plt.plot(data_true[id_result], label = 'true')
                plt.plot(data_pred[id_result], label = 'pred')
                plt.title(id_result)
                plt.legend()
                plt.show()
                
            del model_self_assessment
            
            pass
        pass
    
    ########## SAVE SELF-ASSESSMENT RESULTS ##########
    
    print('\nsave the self-assessment results\n')
    
    path_self_assessment_h5 = os.path.join(path_self_assessment, 'dataset.h5')
    with h5py.File(path_self_assessment_h5, "a") as f:
        
        ###### IMAGES ######
        print('save images')
        
        if 'reconstructed_image' in f.keys():
            d_reconstruction = f['reconstructed_image']
        else:
            d_reconstruction = f.create_group('reconstructed_image')
        
        
        list_tmp = [
            (data_type_tmp, x_hat_tmp)
            for data_type_tmp, x_hat_tmp 
            in zip(['test', 'val', 'train'], [tx_hat, vx_hat, xx_hat]) 
            if data_type_tmp in data_type_list
        ]
        
        for id_data_tmp, data_tmp in list_tmp:
            if not(id_data_tmp in d_reconstruction.keys()):
                d_reconstruction.create_dataset(id_data_tmp, data = data_tmp)
                    
        
        ###### SCORE ######

        print('save score')
        
        for version_sa in all_versions_sa:
            
            if version_sa == 1:
                additional_string_tmp = ''
            else:
                additional_string_tmp = '_v'+str(version_sa)
            
            for data_type_tmp in ['MAE', 'MSE']:
                
                if data_type_tmp+additional_string_tmp in f.keys():
                    d_tmp = f[data_type_tmp+additional_string_tmp]
                else:
                    d_tmp = f.create_group(data_type_tmp+additional_string_tmp)
                    
                
                for score_type_tmp, data_tmp in zip(['predictions_score', 'true_score'], [data_pred, data_true]):
                    
                    if score_type_tmp in d_tmp.keys():
                        d_tmp_score = d_tmp[score_type_tmp]
                    else:
                        d_tmp_score = d_tmp.create_group(score_type_tmp)
                    
                    for data_type in data_type_list:
                
                        id_result = metric_type+'-'+data_type+'-'+str(version_sa)
                        
                        if not(data_type in d_tmp_score.keys()):
                            d_tmp_score.create_dataset(data_type, data = data_tmp[id_result])
            
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
        type=float, 
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
    )

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))