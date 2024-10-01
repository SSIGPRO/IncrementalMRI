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
gamma_sa_MAE_default = 0.001
gamma_sa_MSE_default = 0.00001

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
        all_versions_sa = [1, 2, 3, 4, 5, 6, 7]
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
        flag_evaluate_Dykstra = False
    else:
        flag_evaluate_Dykstra = True
    
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
    
    ########## DATASET ##########
    
    print('load dataset')
    
    (xdata, vdata, tdata) = uty.load_dataset(dataset = dataset)
    
    input_shape = np.shape(xdata[0])
    
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
    
    
    ########## MIRROR STRATEGY ##########
    
    print('mirror strategy')
    
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
    
        ########## MODEL ##########
        
        print('load model')
        
        model = uty.loupe_model(input_shape, 
                                R,
                                dec,
                                L,
                                depth = 5,
                                cartesian_type = cartesian_type,
                               )
    
        ########## MASKER ##########

        print('load masker')
        
        if R_masker != 0:

            kwargs_model_masker = {
                'R': R_masker,
                'R_masker': R_masker_of_the_masker,
            }

            if directory_masker is not None:
                kwargs_model_masker['directory'] = directory_masker
            if additional_id_string_masker is not None:
                kwargs_model_masker['ID'] = additional_id_string_masker

            masker = uty.from_id_to_masker(**kwargs_model_masker)
        else:
            masker = None

        assert masker is not None or R_masker == 0, 'Masker not available. If a masker is not desired, set R_masker = 0.0'

        model_masker = models.add_masker(model, masker = masker)
    
        ########## COMPILE ##########
        
        print('compile model')

        def metric_PSNR(y_true, y_pred):
            return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1))

        def metric_SSIM(y_true, y_pred):
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))

#         def hfen(reconstruction, label, sigma = 1.5):
#             LoG_1 = LoG(reconstruction, sigma = sigma)
#             LoG_2 = LoG(label, sigma = sigma)

#             hfen = ((np.linalg.norm(LoG_1-LoG_2,'fro')**2)/(np.linalg.norm(LoG_2,'fro')**2))**(0.5)
#             return hfen

#         def compute_SSIM_and_HFEN(pdata,
#                                   xdata,
#                                  ):
#             SSIM = []; HFEN = [];

#             for x, p in zip(xdata, pdata):
#                 SSIM += [ssim(p[...,0], x[...,0], data_range = 1)]
#                 HFEN += [hfen(p[...,0], x[...,0])]

#             return [np.array(SSIM), np.array(HFEN)]
        
        # regularization term used in L = {1, 2}
        
        def loss_norm_y(_, y_pred):
            return tf.reduce_mean(tf.norm(y_pred))

        if dec == 0 or dec == 2:
            loss = 'mae'
            loss_weights = [1]
            metrics = [[metric_PSNR, metric_SSIM]]

        elif dec == 1:
            loss = ['mae', loss_norm_y]
            loss_weights = [1-phi, phi]
            metrics = [[metric_PSNR, metric_SSIM], []]

        model_masker.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,),                 
            loss = loss,
            loss_weights = loss_weights,
            metrics = metrics,
        )
        
        ########## LOAD WEIGHTS ##########
        
        if flag_load_weights_model == True:
            
            print('load weights of the model')
            
            ref_file_chpt = [
                'checkpoint.ckpt',
                'checkpoint.ckpt.index',
                'checkpoint.ckpt.data-00000-of-00001',
                'checkpoint',
            ]

            if os.listdir(os.path.dirname(path_checkpoint)) == ref_file_chpt:
                model_masker.load_weights(path_checkpoint)
            else:
                print('model never trained before')
    
    ########## CALLBACK ##########
    
    print('callbacks')
    
    callbacks = uty.callbacks(
        checkpoint = path_checkpoint,
        ldir = path_board,
    )
    
    ########## FIT ##########
    
    if flag_fit_model == True:
        print('fitting the masker model')
        verbose_training = 0
        history = model_masker.fit(xdata,
                                   xdata,
                                   initial_epoch = 0,
                                   epochs= max_epochs_train,
                                   validation_data = (vdata, vdata),
                                   batch_size = batch_size,
                                   callbacks = callbacks,
                                   verbose = verbose_training,
                                  )
    else:
        print('mode operation skips training the model')
    
    ########## CLEAR SESSION ##########
    
    print('clear session')
    
    tf.keras.backend.clear_session()
    
    ########## DYKSTRA ##########
    
    print('Dykstra projection and result')
    with mirrored_strategy.scope():
        iterations_Dykstra = 50

        model_Dykstra = models.add_Dykstra_projection_to_model(
                model_masker, 
                iterations = iterations_Dykstra)

        setting = 'test'
        verbose_settings = False

        model_Dykstra = uty.change_setting(model_Dykstra, 
                                           setting = setting, 
                                           verbose = verbose_settings)

        model_Dykstra.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
                              loss = loss,
                              loss_weights = loss_weights,
                              metrics = metrics,
                             )
        
        if flag_evaluate_Dykstra == True:
            evaluations_Dykstra = model_Dykstra.evaluate(tdata, tdata, batch_size = 10)

    ########## MASK ##########
    
    print('save the mask')
    
    prob_mask = np.array(uty.read_probMask(model_masker))[0,...,0]

    mask = prob_mask > 0.5
    
    path_mask_save = os.path.join(path_mask, 'mask.pkl')
    with open(path_mask_save, 'wb') as f:
        pkl.dump(mask[..., np.newaxis], f)
        
        
    ########## SAVE RECONSTRUCTION RESULTS ##########
    
    
    if flag_evaluate_Dykstra == True:
        
        print('save the masker results')

        path_model_evaluate = os.path.join(path_model, 'model_evaluate.pkl')
        with open(path_model_evaluate, 'wb') as f:
            pkl.dump(evaluations_Dykstra[1], f)
    
    
    ############################## SELF-ASSESSMENT ##############################
    
    ########## PREPARE DATA ##########
    

    def sub_results_for_sa(model_sub_results, data, batch_size = 32, verbose = False):
        
        data_tfdataset = tf.data.Dataset.from_tensor_slices(data)
        data_tfdataset = data_tfdataset.batch(batch_size)
        
        
        [y_diff,
         Finv_y,
         Finv_y_hat,
         x_diff,
         x_hat,
        ] = model_sub_results.predict(data_tfdataset, verbose)

        
        Finv_y = np.array(Finv_y)[..., 0]
        Finv_y_hat = np.array(Finv_y_hat)[..., 0]
        
        x_hat = np.array(x_hat)[0] # comes with a useless dimension (removed here)

        y_diff = np.abs(y_diff[...,0] + 1j*y_diff[..., 1])
        x_diff = np.abs(x_diff[..., 0])

        y_MAE = np.mean(y_diff, (1, 2))
        y_MSE = np.mean(y_diff**2, (1, 2))

        x_MAE = np.mean(x_diff, (1, 2))
        x_MSE = np.mean(x_diff**2, (1, 2))
        
        # return  tdata_sa_MAE, tdata_sa_MSE 
        
        # return (xdata_sa_MAE, ydata_sa_MAE), (xdata_sa_MSE, ydata_sa_MSE)
        
        return (([x_hat, Finv_y, Finv_y_hat, y_MAE], x_MAE),
                ([None,  None,   None,       y_MSE], x_MSE))
    

##########################################################################################################################################
    
    data_type_list = [
        'test',
        'val',
        # 'train',
    ]
    
    if flag_fit_self_assessment == True and not('train' in data_type_list):
        data_type_list += ['train']
    
    with mirrored_strategy.scope():

        model_sub_results = models.model_sub_results_for_self_assessment(
                model_Dykstra, 
                version = 2,
            )

        print('prepare tdata to train self-assessment')
        (([tx_hat, Finv_ty, Finv_ty_hat, ty_MAE], tx_MAE), 
         ([_,      _,       _,           ty_MSE], tx_MSE),
        ) = sub_results_for_sa(model_sub_results, tdata, batch_size)
        
        print('prepare vdata to train self-assessment')
        (([vx_hat, Finv_vy, Finv_vy_hat, vy_MAE], vx_MAE), 
         ([_,      _,       _,           vy_MSE], vx_MSE),
        ) = sub_results_for_sa(model_sub_results, vdata, batch_size)


        print('prepare xdata to train self-assessment')

        if GPUs[0] == '4' or GPUs[0] == '2':
            split_xdata = 2
            batch_size_sub_results =  64
        else:
            split_xdata = 4
            batch_size_sub_results =  32
        
        if dataset == 'IXI':
                
            sub_results = [sub_results_for_sa(model_sub_results,
                                              xdata_tmp, 
                                              batch_size = batch_size_sub_results, 
                                              verbose = True,) 
                           for xdata_tmp in tqdm(np.split(xdata, split_xdata, 0)) ]

        else:
            sub_results = [sub_results_for_sa(
                    model_sub_results,
                    xdata,
                    batch_size = batch_size_sub_results, 
                    verbose = True,
                )]
                    
        xx_hat = np.concatenate(tuple(sub[0][0][0] for sub in sub_results), 0)

        Finv_xy = np.concatenate(tuple(sub[0][0][1] for sub in sub_results), 0)
        Finv_xy_hat = np.concatenate(tuple(sub[0][0][2] for sub in sub_results), 0)
        xy_MAE = np.concatenate(tuple(sub[0][0][3] for sub in sub_results), 0)
        xx_MAE = np.concatenate(tuple(sub[0][1] for sub in sub_results), 0)

        xy_MSE = np.concatenate(tuple(sub[1][0][3] for sub in sub_results), 0)
        xx_MSE = np.concatenate(tuple(sub[1][1] for sub in sub_results), 0)


        ########## CLEAR SESSION ##########
    
    
#     path_self_assessment_h5 = os.path.join(path_self_assessment, 'dataset_to_train_sa.h5')
#     with h5py.File(path_self_assessment_h5, "w") as f:
        
#         if not('dataset' in f.keys()):
#             d = f.create_group('dataset')
#         else:
#             d = f['dataset']
            
#         for d_content, d_name in zip(
#             [
#                 tx_hat, vx_hat, xx_hat,
#                 Finv_ty, Finv_vy, Finv_xy,
#                 Finv_ty_hat, Finv_vy_hat, Finv_xy_hat,
#                 ty_MAE, vy_MAE, xy_MAE,
#                 tx_MAE, vx_MAE, xx_MAE,
#                 ty_MSE, vy_MSE, xy_MSE,
#                 tx_MSE, vx_MSE, xx_MSE,
#             ],[
#                 'tx_hat', 'vx_hat', 'xx_hat',
#                 'Finv_ty', 'Finv_vy', 'Finv_xy',
#                 'Finv_ty_hat', 'Finv_vy_hat', 'Finv_xy_hat',
#                 'ty_MAE', 'vy_MAE', 'xy_MAE',
#                 'tx_MAE', 'vx_MAE', 'xx_MAE',
#                 'ty_MSE', 'vy_MSE', 'xy_MSE',
#                 'tx_MSE', 'vx_MSE', 'xx_MSE',
#             ]):
            
#             if not(d_name in d.keys()):
#                 print('save ',d_name)
#                 d.create_dataset(d_name, data = d_content)
#             else:
#                 print('delete and save ',d_name)
#                 del d[d_name]
#                 d.create_dataset(d_name, data = d_content)
                    
    
     
    
    print('clear session')
    
    del xdata
    del vdata
    del tdata

    tf.keras.backend.clear_session()

    ########## LINEAR SELF-ASSESSMENT ##########

    print('linear self-assessment')

    # train linear regressor
    # line coeff are meant to be a first guess for the neural self-assessment
    # (neural self-assessment will modify them during training)


    regressor_mse = uty.fit_regressor(
        xy_MAE,
        xx_MSE,    
    )
    line_coeff_mse = [regressor_mse.coef_[0], regressor_mse.intercept_]

    regressor_mae = uty.fit_regressor(
        xy_MAE,
        xx_MAE,    
    )
    line_coeff_mae = [regressor_mae.coef_[0], regressor_mae.intercept_]
    
    
    data_pred = {}
    data_true = {}
    
    
    for version_sa in all_versions_sa:
        
        print('----------- VERSION S-A -----------\n\t\t',version_sa)
        
        tf.keras.backend.clear_session()
        
        for (metric_type, 
             ty, vy, xy, 
             tx, vx, xx, 
             line_coeff) in zip(
                    ['MAE', 'MSE'],
                    [ty_MAE, ty_MAE], [vy_MAE, vy_MAE], [xy_MAE, xy_MAE],
                    [tx_MAE, tx_MSE], [vx_MAE, vx_MSE], [xx_MAE, xx_MSE],
                    [line_coeff_mae, line_coeff_mse],
                    # ['MSE'],
                    # [ty_MAE], [vy_MAE], [xy_MAE],
                    # [tx_MSE], [vx_MSE], [xx_MSE],
                    # [line_coeff_mse],
                ):
        
            print('------ METRIC TYPE -----\n\t',metric_type)
                
            ########## DATASET FOR SA #########

            print('create dataset for sa')

            if version_sa in [1,3,8,9,11, 101,102,103,104,105,106]:
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment,] = [
                        ([x_hat, y], x) 
                    for x_hat, y, x in zip(
                            [tx_hat, vx_hat, xx_hat],
                            [ty, vy, xy],
                            [tx, vx, xx],     
                        )]

            elif version_sa in [2,4]:

                Finv_ty_diff = Finv_ty - Finv_ty_hat
                Finv_vy_diff = Finv_vy - Finv_vy_hat
                Finv_xy_diff = Finv_xy - Finv_xy_hat
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment] = [
                        ([np.concatenate((x_hat, Finv_y_diff[..., np.newaxis]), 2),
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
            
            elif version_sa in [5,6]:
                
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment] = [
                        (np.concatenate((x_hat,
                                         Finv_y[..., np.newaxis], 
                                         Finv_y_hat[..., np.newaxis]), 
                                        -1),
                         x
                        )
                    for x_hat, Finv_y, Finv_y_hat, x in zip(
                            [tx_hat, vx_hat, xx_hat],
                            [Finv_ty, Finv_vy, Finv_xy],
                            [Finv_ty_hat, Finv_vy_hat, Finv_xy_hat],
                            [tx, vx, xx],     
                        )
                ]
                pass
            
            elif version_sa in [7]:
                # print(np.shape(x_hat))
                # print(np.shape(Finv_y))
                # print(np.shape(Finv_y_hat))
                [tdata_self_assessment,
                 vdata_self_assessment,
                 xdata_self_assessment] = [
                        ([np.concatenate((x_hat,
                                          Finv_y[..., np.newaxis], 
                                          Finv_y_hat[..., np.newaxis]), 
                                         -1),
                          y],
                         x
                        )
                
                    for x_hat, Finv_y, Finv_y_hat, y, x in zip(
                            [tx_hat, vx_hat, xx_hat],
                            [Finv_ty, Finv_vy, Finv_xy],
                            [Finv_ty_hat, Finv_vy_hat, Finv_xy_hat],
                            [ty, vy, xy],
                            [tx, vx, xx],     
                        )
                ]
                # print(np.shape(tdata_self_assessment[0]))
                pass

            ########## MODELS ##########

            
            print('retrieve models self-assessment')

            with mirrored_strategy.scope():

                if version_sa in [1,2,3,4,7,8,9,11, 101,102,103,104,105,106]:
                    if version_sa in [2,4,7,8]:
                        filt = [40,20,10,8,6,4,2,1]
                        kern = [5]*len(filt)
                        pool_size = [(2,2)]*len(filt)
                        average_pooling = False
                        poly_degree = 1
                        dense = False
                        conv_per_block_image = 1

                    elif version_sa in [1,3,102,103,104,106]:
                        filt = [10,8,6,4,2,1]
                        kern = [5]*len(filt)
                        pool_size = [(2,2)]*len(filt)
                        average_pooling = True
                        poly_degree = 2
                        dense = False
                        conv_per_block_image = 1
                        units = [100, 20]
                        
                    elif version_sa in [105]:
                        filt = [10,8,6,4,2,1]
                        kern = [5]*len(filt)
                        pool_size = [(2,2)]*len(filt)
                        average_pooling = True
                        poly_degree = 2
                        dense = False
                        conv_per_block_image = 1
                        units = [10]
                        
                    elif version_sa in [101]:
                        filt = [10,8,6,4,2,1]
                        kern = [5]*len(filt)
                        pool_size = [(2,2)]*len(filt)
                        average_pooling = True
                        poly_degree = 2
                        dense = False
                        conv_per_block_image = 2
                        
                    elif version_sa in [9]:
                        filt = [40,20,10,8,6]
                        kern = [3]*len(filt)
                        pool_size = [(2,2)]*len(filt)
                        average_pooling = False
                        poly_degree = 1
                        dense = True
                        conv_per_block_image = 2
                        
                    elif version_sa in [11]:
                        filt = [2,4,8,16,32,64]
                        kern = [3]*len(filt)
                        pool_size = [(3,3)]*len(filt)
                        average_pooling = False
                        poly_degree = 1
                        dense = True
                        conv_per_block_image = 2

                
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
                    )
                    
                    
                elif version_sa in [5,6]:
                    
                    depth = 3 * 9 + 2
                    
                    model_self_assessment = models.resnet(
                        np.shape(xdata_self_assessment[0])[1:],
                        depth,
                    )
                    

                ########## COMPILE ##########

                def mae_with_penalty(y_true, y_pred, gamma = 2):

                    diff = y_true-y_pred

                    mae = tf.math.reduce_mean(tf.math.abs(diff))

                    mae = mae * (
                            1 + gamma * 
                            tf.math.reduce_mean(tf.cast(
                                tf.math.greater(y_true, y_pred), tf.float32))
                        )
                    
                    return mae
                
                def mae_with_regularization(y_true, y_pred, gamma = 0.001):
                    
                    diff = y_true-y_pred

                    mae = tf.math.reduce_mean(tf.math.abs(diff))
                    
                    mae = (1-gamma) * mae + gamma * tf.math.reduce_mean(tf.cast(
                        tf.math.greater(y_true, y_pred), tf.float32))

                    return mae
                    

                def mse_with_penalty(y_true, y_pred, gamma = 2):

                    diff = y_true-y_pred

                    mse = tf.math.reduce_mean(tf.math.pow(diff, 2))

                    mse = mse * (
                            1 + gamma * 
                            tf.math.reduce_mean(tf.cast(
                                tf.math.greater(y_true, y_pred), tf.float32))
                        )
                    
                    return mse
                
                def mse_with_regularization(y_true, y_pred, gamma = 0.001):
                    
                    diff = y_true-y_pred

                    mse = tf.math.reduce_mean(tf.math.abs(diff))
                    
                    mse = (1-gamma) * mse + gamma * tf.math.reduce_mean(tf.cast(
                        tf.math.greater(y_true, y_pred), tf.float32))

                    return mse

                # CHANGE SA LOSS

                if version_sa == [3,4]:
                    if metric_type == 'MAE':
                        loss = lambda y_true, y_pred: mae_with_penalty(
                            y_true, y_pred, gamma = gamma_sa_MAE)
                    elif metric_type == 'MSE':
                        loss = lambda y_true, y_pred: mse_with_penalty(
                            y_true, y_pred, gamma = gamma_sa_MSE)
                        
                elif version_sa in [2,5,11]:

                    if metric_type == 'MAE':
                        loss = 'mae'
                    elif metric_type == 'MSE':
                        loss = 'mse'
                        
                elif version_sa in [1, 101,102,103,104,105]:
                    loss = 'mae'
                    
                elif version_sa in [106]:
                    
                    if metric_type == 'MAE':
                        loss = lambda y_true, y_pred: mae_with_regularization(
                            y_true, y_pred, gamma = gamma_sa_MAE)
                    elif metric_type == 'MSE':
                        loss = lambda y_true, y_pred: mae_with_regularization(
                            y_true, y_pred, gamma = gamma_sa_MSE)
                        
                elif version_sa in [6,7,8]:
                    
                    if metric_type == 'MAE':
                        loss = lambda y_true, y_pred: mse_with_regularization(
                            y_true, y_pred, gamma = gamma_sa_MAE)
                    elif metric_type == 'MSE':
                        loss = lambda y_true, y_pred: mse_with_regularization(
                            y_true, y_pred, gamma = gamma_sa_MSE)
                        
                elif version_sa in [9]:
                    if metric_type == 'MAE':
                        loss = lambda y_true, y_pred: mae_with_regularization(
                            y_true, y_pred, gamma = gamma_sa_MAE)
                    elif metric_type == 'MSE':
                        loss = lambda y_true, y_pred: mse_with_regularization(
                            y_true, y_pred, gamma = gamma_sa_MSE)
                    
                
                print('compile self-assessment')

                model_self_assessment.compile(
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_self_assessment),
                    loss = loss,
                )

                
            ########## PATH ##########

            # add str for self assessment
            if version_sa in [3,4,6,7,8,9]:
                if metric_type == 'MAE':
                    gamma_sa_tmp = gamma_sa_MAE
                elif metric_type == 'MSE':
                    gamma_sa_tmp = gamma_sa_MSE
            else:
                gamma_sa_tmp = None
            
            print('path self-assessment')
            paths, existance_path_tmp = uty.from_self_assessment_to_useful_paths(
                path_self_assessment, 
                create_path = create_path_sa,
                version_sa = version_sa,
                gamma_sa = gamma_sa_tmp,
            )
            
            if metric_type == 'MAE':
                path_self_assessment_checkpoint = paths[1]
                path_self_assessment_board = paths[2]
            elif metric_type == 'MSE':
                path_self_assessment_checkpoint = paths[4]
                path_self_assessment_board = paths[5]

            ########## LOAD WEIGHTS ##########

            if flag_load_weights_self_assessment == True and existance_path_tmp == True:

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
                    print('self-assessment never trained before')
                    
            if version_sa in [103,104,105,106]:
                
                paths_v1, existance_path_tmp_v1 = uty.from_self_assessment_to_useful_paths(
                    path_self_assessment, 
                    create_path = create_path_sa,
                    version_sa = 1,
                    gamma_sa = gamma_sa_tmp,
                )

                if metric_type == 'MAE':
                    path_self_assessment_checkpoint_v1 = paths_v1[1]
                elif metric_type == 'MSE':
                    path_self_assessment_checkpoint_v1 = paths_v1[4]
                
                ref_file_chpt = [
                    'checkpoint.ckpt',
                    'checkpoint.ckpt.index',
                    'checkpoint.ckpt.data-00000-of-00001',
                    'checkpoint',
                ]

                if os.listdir(
                    os.path.dirname(
                        path_self_assessment_checkpoint_v1)) == ref_file_chpt:

                    print('load V1 weights self-assessment')
                    model_self_assessment.load_weights(
                        path_self_assessment_checkpoint_v1)
                else:
                    print('DID NOT load V1 weights self-assessment')
               
                    
                with mirrored_strategy.scope():

                    # inputs

                    input_image = tf.keras.Input(
                            model_self_assessment.inputs[0].shape[1:], 
                            name = 'input_image',
                        )
                    input_metric = tf.keras.Input(
                            model_self_assessment.inputs[1].shape[1:], 
                            name = 'input_metric',
                        )

                    # model_conv

                    model_image = tf.keras.Model(
                        inputs = model_self_assessment.inputs[0],
                        outputs = model_self_assessment.get_layer('batch_norm_5').output,
                        name = 'sub_model_conv',
                    )

                    feature_image = model_image(input_image)

                    # model_dense

                    first_image = tf.keras.Input(model_image.output.shape[1:])

                    out_image = tf.keras.layers.Flatten()(first_image)
                    
                    for u in units:
                        out_image = tf.keras.layers.Dense(u, activation='relu')(out_image)
                        out_image = tf.keras.layers.BatchNormalization()(out_image)
                    
                    # out_image = tf.keras.layers.Dense(100, activation='relu')(out_image)
                    # out_image = tf.keras.layers.BatchNormalization()(out_image)
                    # out_image = tf.keras.layers.Dense(20, activation='relu')(out_image)
                    # out_image = tf.keras.layers.BatchNormalization()(out_image)
                    out_image = tf.keras.layers.Dense(1, activation='relu')(out_image)
                    out_image = tf.keras.layers.BatchNormalization()(out_image)

                    model_dense = tf.keras.Model(
                        inputs = first_image, 
                        outputs = out_image,
                        name = 'sub_model_dense',
                    )

                    out_image = model_dense(feature_image)

                    # model_metric

                    model_metric = tf.keras.Model(
                        inputs = model_self_assessment.inputs[1],
                        outputs = model_self_assessment.get_layer('linear_reg_dense').output,
                        name = 'sub_model_metric',
                    )

                    out_metric = model_metric(input_metric)

                    # model_correction

                    model_correction = tf.keras.Model(
                        inputs = model_self_assessment.get_layer('concat').input,
                        outputs = model_self_assessment.outputs,
                        name = 'sub_model_correction',
                    )

                    out = model_correction([out_image, out_metric])

                    model_self_assessment = tf.keras.Model(
                        inputs = [input_image, input_metric],
                        outputs = out,
                        name = 'model_self_assessment_version'+str(version_sa),
                    )

                    for i in model_self_assessment.layers:
                        if 'sub_model_dense' in i.name:
                            i.trainable = True

                        if 'sub_model_conv' in i.name:
                            i.trainable = False

                        if 'sub_model_correction' in i.name:
                            i.trainable = False

                        model_self_assessment.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_self_assessment),
                            loss = loss,
                        )
                
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
                
                if version_sa in [104,105]:
                    
                    model_self_assessment.optimizer.learning_rate = lr_self_assessment/25
                    
                    for i in model_self_assessment.layers:
                        if 'sub_model_dense' in i.name:
                            i.trainable = True

                        if 'sub_model_conv' in i.name:
                            i.trainable = True

                        if 'sub_model_correction' in i.name:
                            i.trainable = True
                    
                    history = model_self_assessment.fit(
                        x = xdata_self_assessment[0],
                        y = xdata_self_assessment[1],
                        initial_epoch = len(history.history['loss']),
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

            del model_self_assessment
            
            pass
        pass
    
    ########## SAVE SELF-ASSESSMENT RESULTS ##########
    
    print('\nsave the self-assessment results\n')
    
    path_self_assessment_h5 = os.path.join(path_self_assessment, 'dataset.h5')
    print(path_self_assessment_h5)
    
    with h5py.File(path_self_assessment_h5, "a") as f:
        
        ###### IMAGES ######
        print('save images')
        
        # if not(d_name in d.keys()):
        #         print('create ',d_name)
        #         d.create_dataset(d_name, data = d_content)
        #     else:
        #         print('delete and create ',d_name)
        #         del d[d_name]
        #         d.create_dataset(d_name, data = d_content)
        
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
                print('create dataset ', id_data_tmp)
                d_reconstruction.create_dataset(id_data_tmp, data = data_tmp)
            else:
                print('delete and create ', id_data_tmp)
                del d_reconstruction[id_data_tmp]
                d_reconstruction.create_dataset(id_data_tmp, data = data_tmp)
                
        ###### SCORE ######

        print('save score')
        
        for version_sa in all_versions_sa:
            
            for metric_type, gamma_sa_tmp in zip(['MAE', 'MSE'], [gamma_sa_MAE, gamma_sa_MSE]):
            # for metric_type, gamma_sa_tmp in zip(['MSE'], [gamma_sa_MSE]):
                
                if version_sa == 1:
                    additional_string_tmp = ''
                else:
                    additional_string_tmp = '_v'+str(version_sa)

                if version_sa in [3,4,6,7,8,9]:
                    if gamma_sa_tmp != 2:
                        additional_string_tmp = additional_string_tmp + '_g'+str(gamma_sa_tmp)
                
                
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
                        d_tmp_score = d_tmp[score_type_tmp]
                    else:
                        d_tmp_score = d_tmp.create_group(score_type_tmp)
                    
                    for data_type in data_type_list:
                
                        id_result = metric_type+'-'+data_type+'-'+str(version_sa)
                        print(id_result)
                        if not(data_type in d_tmp_score.keys()):
                            d_tmp_score.create_dataset(data_type, data = data_tmp[id_result])
                        else:
                            del d_tmp_score[data_type]
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
    )

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))