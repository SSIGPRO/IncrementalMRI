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
from tensorflow.image import psnr as PSNR
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_laplace as LoG
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
        
    
    def hfen(label, reconstruction, sigma = 1.5):
        LoG_1 = LoG(reconstruction, sigma = sigma)
        LoG_2 = LoG(label, sigma = sigma)

        hfen = ((np.linalg.norm(LoG_1-LoG_2,'fro')**2)/(np.linalg.norm(LoG_2,'fro')**2))**(0.5)

        return hfen

    def compute_SSIM_and_HFEN(pdata,
                              xdata,
                             ):
        SSIM = []; HFEN = [];

        for x, p in zip(xdata, pdata):
            SSIM += [ssim(x[...,0], p[...,0], data_range = 1)]
            HFEN += [hfen(x[...,0], p[...,0])]

        return [np.array(SSIM), np.array(HFEN)]
        
    print('\ncompute SSIM and HFEN')
    
    p_xdata = model_Dykstra.predict(xdata, batch_size = batch_size)
    print(np.shape(xdata), np.shape(p_xdata))
    
    [SSIM_train, HFEN_train] = compute_SSIM_and_HFEN(xdata, model_Dykstra.predict(xdata, batch_size = batch_size))
    [SSIM_val, HFEN_val] = compute_SSIM_and_HFEN(vdata, model_Dykstra.predict(vdata, batch_size = batch_size))
    [SSIM_test, HFEN_test] = compute_SSIM_and_HFEN(tdata, model_Dykstra.predict(tdata, batch_size = batch_size))
    print('done\n')
    
    ########## SAVE SELF-ASSESSMENT RESULTS ##########
    
    print('\nsave the self-assessment results\n')
    
    path_self_assessment_h5 = os.path.join(path_self_assessment, 'dataset.h5')
    print(path_self_assessment_h5)
    
    with h5py.File(path_self_assessment_h5, "a") as f:
        
        ##### HFEN and SSIM ####
        
        print('hfen_ssim-R_'+str(R))
        
        if 'hfen_ssim-R_'+str(R) in f.keys():
            d_hfen_ssim = f['hfen_ssim-R_'+str(R)]
        else:
            d_hfen_ssim = f.create_group('hfen_ssim-R_'+str(R))
        
        for data_id, data in zip(
                ['HFEN_true', 'SSIM_true'],
                [
                    [HFEN_train, HFEN_val, HFEN_test,],
                    [SSIM_train, SSIM_val, SSIM_test,],
                ],
            ):
            
            for data_tmp, data_type in zip(
                    data,
                    ['train', 'val', 'test',],
                ):
                
                id_data_tmp = data_id+'_'+data_type
                
                if not(id_data_tmp in d_hfen_ssim.keys()):
                    print('create dataset ', id_data_tmp)
                    d_hfen_ssim.create_dataset(id_data_tmp, data = data_tmp)
                else:
                    print('delete and create ', id_data_tmp)
                    del d_hfen_ssim[id_data_tmp]
                    d_hfen_ssim.create_dataset(id_data_tmp, data = data_tmp)
    
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