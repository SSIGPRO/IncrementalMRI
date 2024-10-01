import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os
from modules import models
import tensorflow.keras.backend as K

import pickle as pkl
from tqdm import tqdm
import h5py
import itertools


################ Path Menager ################


def from_id_to_model_path(create_path = False, **kwargs):
    
    path = os.path.join('save_models')
    if os.path.isdir(path) == False:
        os.mkdir(path)
    
    if 'directory' in kwargs:
        directory = kwargs.pop('directory')
        
        path = os.path.join(path, directory)
        if os.path.isdir(path) == False:
            os.mkdir(path)
    
    kw_vals = list(kwargs.values())
    kw_keys = list(kwargs.keys())
    
    id_string = '-'.join([k+'_'+str(v) for k, v in zip(kw_keys, kw_vals)])
    
    path = os.path.join(path,  id_string)
    
    if os.path.isdir(path) == False:
    
        if create_path == True:
            os.mkdir(path)
            existance = True
        else:
            existance = False
    else:
        existance = True
        
    return path, existance


def from_model_path_to_useful_paths(model_path, create_path = True):
    
    folders = ['checkpoint', os.path.join('checkpoint','checkpoint.ckpt'), 'board', 'mask', 'self-assessment']
    
    path_folders = [os.path.join(model_path, f) for f in folders]
    
    existance = [os.path.isdir(f) for f in path_folders]
    
    if create_path == True:
        [os.mkdir(f) for e, f in zip(existance, path_folders) if 
         e == False]
        existance = [True]*len(folders)
    
    existance = (existance == [True]*len(folders))
    
    return path_folders[1:], existance


def from_id_to_masker(**kwargs):
    
    path, existance = from_id_to_model_path(create_path = False, **kwargs)
    
    assert existance == True, f'model of the masker not found in {path}'
    
    [_, _, path_masker, _], existance = from_model_path_to_useful_paths(path, create_path = False)

    path_masker = os.path.join(path_masker, 'mask.pkl')
        
    with open(path_masker, 'rb') as f:
        masker = pkl.load(f)

    return masker


def from_self_assessment_to_useful_paths(
        path_self_assessment,
        create_path = True,
        version_sa = None,
        gamma_sa = None,
        k_fold_ind = None,
        threshold = None,
        metric_type_list = ['MAE', 'MSE'],
    ):
    
    if version_sa == 1 or version_sa is None:
        additional_string = ''
    else:
        additional_string = '_v'+str(version_sa)
        
    if gamma_sa is not None and gamma_sa != 2:
        additional_string = additional_string + '_g'+str(gamma_sa)
        
    if k_fold_ind is not None:
        additional_string = additional_string + '-k_fold_'+str(k_fold_ind)
        
    if threshold is not None:
        additional_string = additional_string + '-threshold_'+str(threshold)
        
    tmp_path = [m+additional_string for m in metric_type_list]
    
    # tmp_path = ['MAE'+additional_string, 'MSE'+additional_string, 'SSIM'+additional_string, ]
    tmp_path_nested = ['checkpoint',
                       os.path.join('checkpoint','checkpoint.ckpt'),
                       'board',
                      ]
    paths = []
    existance = []
    for tmp in tmp_path:
        
        path_tmp = os.path.join(path_self_assessment, tmp)
        if not(os.path.isdir(path_tmp)) and create_path == True:
            os.mkdir(path_tmp)
            
        for tmp_nested in tmp_path_nested:
            path_tmp_nested = os.path.join(path_tmp, tmp_nested)
            if not(os.path.isdir(path_tmp_nested)):
                if create_path == True:
                    os.mkdir(path_tmp_nested)
                    existance += [True]
                else:
                    existance += [False]
            else:
                existance += [True]
                
            paths += [path_tmp_nested]
    
    return paths, existance

################ General Utilities ################

def load_dataset(dataset='PD'):
    
    if dataset == 'PD' or dataset == 'PDFS':
        
        # trainDataPath = os.path.join('..','..','datasets','kneeMRI','fastMRI_trainSet_esc_320_'+dataset+'.hdf5') 
        # valDataPath = os.path.join('..','..','datasets','kneeMRI','fastMRI_valSet_esc_320_'+dataset+'.hdf5')
        # testDataPath = os.path.join('..','..','datasets','kneeMRI','fastMRI_testSet_esc_320_'+dataset+'.hdf5')
        
        trainDataPath = os.path.join('..','datasets','kneeMRI','fastMRI_trainSet_esc_320_'+dataset+'.hdf5') 
        valDataPath = os.path.join('..','datasets','kneeMRI','fastMRI_valSet_esc_320_'+dataset+'.hdf5')
        testDataPath = os.path.join('..','datasets','kneeMRI','fastMRI_testSet_esc_320_'+dataset+'.hdf5')

        xdata = loadData(trainDataPath) # images are already rescaled: 1 is the max value of the whole volume
        vdata = loadData(valDataPath) # already split validation set
        tdata = loadData(testDataPath)
        

    elif dataset == 'IXI':
        def find_reshape_shape(x):
            old_shape = np.shape(x)
            new_shape = (int(old_shape[0]*old_shape[1]), ) + old_shape[2:] + (1, )
            return new_shape
        path_dataset = os.path.join('..','datasets','IXI-T1-dataset','h5','dataset.h5')
        with h5py.File(path_dataset, 'r') as h5:
            xdata = np.array(h5.get('xdata'))
            vdata = np.array(h5.get('vdata'))
            tdata = np.array(h5.get('tdata'))
        xdata = np.reshape(xdata, find_reshape_shape(xdata))
        vdata = np.reshape(vdata, find_reshape_shape(vdata))
        tdata = np.reshape(tdata, find_reshape_shape(tdata))
    return (xdata, vdata, tdata)


def load_sub_dataset(dataset='PD', sub = ['train', 'val', 'test']):
    
    all_sub = ['train', 'val', 'test']
    
    bools = [s in sub for s in all_sub]
    
    data = []
    
    if dataset == 'PD' or 'PDFS':

        for i in itertools.compress(all_sub, bools):
            path = os.path.join('..','..','datasets','kneeMRI','fastMRI_'+i+'Set_esc_320_'+dataset+'.hdf5') 
            data += loadData(path)
        
    elif dataset == 'IXI':
        def find_reshape_shape(x):
            old_shape = np.shape(x)
            new_shape = (int(old_shape[0]*old_shape[1]), ) + old_shape[2:] + (1, )
            return new_shape
        
        
        path = os.path.join('..','..','datasets','IXI-T1-dataset','h5','dataset.h5')
        with h5py.File(path_dataset, 'r') as h5:

            for i in itertools.compress(['x', 'v', 't'], bools):
                data_tmp = np.array(h5.get(i+'data'))
                shape = find_reshape_shape(data_tmp)
                data += np.reshape(data_tmp, shape)
            
    return data


def loadData(data_path): # load from the given folder the h5py dataset
    with h5py.File(data_path, 'r') as f:
        xdata = f[os.path.join('dataset')] [()]
    return xdata  

def _rotate_corners_mask(mask, plot = False, ax = None):
    """
        Splits the image in 4 windows and rotates them by 180°.
        
        This enambles to visualize the mask coherentely
        with our paper and with the original paper or LOUPE.
    """
    
    def rot180(inpu):
        output = np.rot90(np.rot90(inpu))
        return output
    
    row = int(np.shape(mask)[0]/2)
    col = int(np.shape(mask)[1]/2)

    ul = mask[:row,:col] # upper left
    ur = mask[:row,col:] # upper right
    bl = mask[row:,:col] # below left
    br = mask[row:,col:] # below right
    
    ul_rotated = rot180(ul)
    ur_rotated = rot180(ur)
    bl_rotated = rot180(bl)
    br_rotated = rot180(br)

    u = np.append(ul_rotated, ur_rotated, 1)
    b = np.append(bl_rotated, br_rotated, 1)
    mask_rotated = np.append(u,b,0)
    
    if plot == True:
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(mask_rotated, vmin = 0, vmax = 1, cmap = 'gray')
        ax.title.set_text('mask')
        ax.axis('off')
    return mask_rotated

def find_best_model_id(
        R, 
        R_masker, 
        max_model_ID,
        directory,
        verbose = False,
    ):

    PSNR = 0
    additional_id_string = None

    for additional_id_string_tmp in range(max_model_ID+1):

        kwargs_model_tmp = {
            'R': R,
            'R_masker': R_masker,
        }

        if directory is not(None):
            kwargs_model_tmp['directory'] = directory
        if additional_id_string_tmp is not None:
            kwargs_model_tmp['ID'] = additional_id_string_tmp

        path_model_tmp, _ = from_id_to_model_path(**kwargs_model_tmp)

        with open(os.path.join(path_model_tmp, 'model_evaluate.pkl'), 'rb') as f:
            PSNR_tmp = pkl.load(f)

        if PSNR_tmp > PSNR:
            PSNR = PSNR_tmp
            additional_id_string = additional_id_string_tmp

    if verbose == True:
        print('\nBest ID: ',additional_id_string,
              '\nBest PSNR = ', PSNR,'\n')
    return additional_id_string

def handle_GPUs(GPUs = None, enable_GPU=1,):
    """
        For the specified GPUs, "memory growth" is activated.
        The un-specified GPUs are turned invisible to the kernel.
        
        GPUs is a string contraining the number of GPUs (e.g., GPUs = '0,1')
        If GPUs == None all GPUs are activated.
    """
    
    if GPUs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= GPUs
    
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for _gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(_gpu, enable_GPU)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print('number of Logical GPUs:', len(logical_gpus))
    pass

def callbacks(ldir = None,   
              checkpoint = None,   
              monitor='val_loss',   
              patienceEarlyStop=100,  
              patienceLR=50,  
              min_lr = 0.00001,  
              min_delta = 0.00000001,  
              redureLR = 0.2,  
              verbose = 0,  
             ):
    
    callback_list=[
        tf.keras.callbacks.TerminateOnNaN(),       
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor,
                                             factor = redureLR,
                                             patience = patienceLR,
                                             min_lr = min_lr,
                                             min_delta=min_delta,
                                             verbose = verbose,
                                            ),
      
        tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                         min_delta=min_delta,
                                         patience=patienceEarlyStop,
                                         verbose=verbose,
                                         mode='auto',
                                         baseline=None,
                                         restore_best_weights=True,
                                        ),                 
    ]
    
    if ldir != None:
        callback_list += [        
            tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint,
                                               monitor = monitor,
                                               verbose = verbose,
                                               save_best_only = True,
                                               save_weights_only = True,
                                               save_freq = 'epoch',
                                              ),
        ]
    if checkpoint != None:
        callback_list += [
            tf.keras.callbacks.TensorBoard(log_dir=ldir,
                                           histogram_freq=0,
                                           write_graph=False,
                                           write_images=False,
                                           update_freq='epoch',
                                           embeddings_freq=0,
                                           embeddings_metadata=None,
                                          ),
        ]
    return callback_list

def copy_weights_by_matching_layer_name(model_dest, model, verbose = False):
    """
        It copies all the weights of the layers sharing the same name 
        from model to model_dest
    """
    
    for layer_dest in model_dest.layers:
        for layer in model.layers:
            if layer_dest.name == layer.name:
                layer_dest.set_weights(layer.get_weights())
                if verbose==True:
                    print(layer_dest.name)
                pass
            pass
        pass
    return model_dest


def loupe_model(input_shape, R, dec, L = 0, depth = 5,
                name = None, cartesian_type = None, 
                fixed_mask_from_outside = False, mask = None,
               ):
    """
        It returns the selected LOUPE between:
        dec0|L0 (original LOUPE)
        dec1|L1
        dec1|L2
        dec2|L0
        
        input_shape is the shape of the generic image the network 
        takes as input (e.g., (320, 320, 1))
        
        R is the speed-up factor
        
        To create deci*|Lj (for a general i and j) run the function 
        "model = add_Dykstra_projection_to_model(model)" 
        from "models.py" (see the demo for an example)
    """
    
    assert dec in [0, 1, 2], '"dec" must be "0, 1, 2"'
    if dec == 0:
        model = models.dec0(input_shape = input_shape,
                            R = R,
                            depth = depth,
                            name = name,
                           )
    elif dec == 1:
        model = models.dec1(input_shape = input_shape,
                            R = R,
                            L = L,
                            depth = depth,
                            name = name,
                          )
    elif dec == 2:
        model = models.dec2(input_shape = input_shape,
                            R = R,
                            depth = depth,
                            name = name,
                            cartesian_type = cartesian_type,
                            fixed_mask_from_outside = fixed_mask_from_outside, 
                            mask = mask,
                           )
    
    model = set_slope_trainability(model, trainable = False)
    
    return model

############### MODIFY LAYER TRAINABILITY ###############

def set_slope_trainability(model, trainable = False, verbose = False):
    """
        sets the slope trainability to "trainable"
        of all the layers "sampled_mask" of the model
    """
    
    for i, l in enumerate(model.layers):
        if l.name=='sampled_mask':
            if verbose == True:
                print('\n\n\t\t SET SLOPE TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.set_attributes(trainable = trainable)
            if verbose == True:
                print('- after: ', l.trainable)
    return model

def set_neurons_trainability(model, trainable, verbose=False):
    """
        sets the trainability of the decoder
    """
    for i, l in enumerate(model.layers):
        if l.name.find('mask')==-1:
            if verbose==True:
                print('\n\n\t\tSET NEURONS TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.set_attributes(trainable = trainable)
            if verbose==True:
                print('- after: ', l.trainable)
    return model


def set_probMask_trainability(model, trainable, verbose=False):
    """
        sets the the trainability of the encoder
    """
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask')!=-1 and l.name.find('prob_mask_')==-1:
            if verbose==True:
                print('\n\n\t\tSET PROB MASK TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.set_attributes(trainable = trainable)
            if verbose==True:
                print('- after: ', l.trainable)
    return model


def set_mask_slope(model, slope, verbose=False):
    """
        updates the slope value 
        of all the layers "sampled_mask" of the model
    """
    for i, l in enumerate(model.layers):
        if l.name.find('sampled_mask')!=-1:
            if verbose==True:
                print('\n\n\t\SET MASK SLOPE')
                print('layer n°: ', i,' - ', l.name, ' - slope:')
                print('- before : ', l.slope)
            l.set_attributes(slope = slope)
            if verbose==True:
                print('- after: ', l.slope)
    return model

def set_mask_hard_threshold(model, hard_threshold, verbose=False):
    """
        makes the model binarize (or not) the mask before using it
    """
    for i, l in enumerate(model.layers):
        if l.name.find('undersample')!=-1:
            if verbose==True:
                print('\n\n\t\tSET MASK THRESHOLD')
                print('layer n°: ', i,' - ', l.name, ' - hard_threshold:')
                print('- before : ', l.hard_threshold)
            l.set_attributes(hard_threshold = hard_threshold)
            if verbose==True:
                print('- after: ', l.hard_threshold)
    return model

def set_mask_randomicity(model, randomicity = True, verbose=False):
    """
       Fixes (or unfixes) the mask of the model 
    """
    if randomicity==True:
        maxval = 1.0
        minval = 0.0
    else:
        maxval = 0.50000001
        minval = 0.49999999
        
        
    for i, l in enumerate(model.layers):
        if l.name.find('random_mask')!=-1:
            if verbose==True:
                print('\n\n\t\tSET MASK RANDOMICITY')
                print('layer n°: ', i,' - ', l.name, ' - randomicity:')
                print('- before : ', l.maxval, l.minval)
            l.set_attributes(maxval = maxval, minval = minval)
            if verbose==True:
                print('- after: ', l.maxval, l.minval)
    return model

def set_mask_R(model, R, verbose=False):
    """
        updates the speed-up factore of all the masks of the model
    """
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask_scaled')!=-1:
            if verbose==True:
                print('\n\n\t\tSET MASK R')
                print('layer n°: ', i,' - ', l.name, ' - R:')
                print('- before : ', l.R)
            l.set_attributes(R = R)
            if verbose==True:
                print('- after: ', l.R)
    return model

def write_probMask(model, probMask, verbose = False):
    """
        updates the probability mask
    """
    j = 0
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask')!=-1 and l.name.find('prob_mask_scaled')==-1:
            if verbose==True:
                print('layer n°: ', i,' - ', l.name)
            l.write_probMask(probMask[j])
            j = j + 1
            if verbose==True:
                print('mask has been written')
    return model

def read_probMask(model, verbose = False, return_gamma = False, ):
    probMask = []
    for i, l in enumerate(model.layers):
        if l.name.find('prob_mask')!=-1 and l.name.find('prob_mask_scaled')==-1:
            if verbose==True:
                print('layer n°: ', i,' - ', l.name)
            probMask += [l.read_probMask(return_gamma = return_gamma)]
            if verbose==True:
                print('mask has been read')
    return probMask

def change_setting(model, setting = 'test', verbose = False):
    
    """
        Switchs the selected model from "train" to "test" mode 
        (or viceversa) by binarizing and fixing the mask.
        To fix the mask, the random number generator layer
        is tuned to only generate deterministic numbers (0.5)
        To binarize the mask the threshold operation (>0.5) is added.
    """
    
    assert setting == 'test' or setting == 'train', 'setting must be "test" or "setting"'
    
    if setting == 'test':
        randomicity = False
        hard_threshold = True
    elif setting == 'train':
        randomicity = True
        hard_threshold = False
        
    model = set_mask_randomicity(model, randomicity = randomicity, 
                                 verbose = verbose)
    model = set_mask_hard_threshold(model, hard_threshold = hard_threshold,
                                    verbose=verbose)
    
    return model

############### SELF-ASSESSMENT ###############

def find_decremental_speedup(score_true_list, 
                             score_pred_list, 
                             R_list, 
                             threshold, 
                             metric_type = 'max_best',
                            ):
    
    
    shape_reference = np.shape(score_true_list[0])
    speedup_mask = np.array(np.zeros(shape_reference)>1)
    speedup_R = np.zeros(shape_reference)
    speedup_metric = np.zeros(shape_reference)
    
    
    R_list = np.sort(R_list)
    
    if metric_type == 'max_best':
        R_list = R_list[::-1]
        
    argsort = np.argsort(R_list)
    
    R_list = np.array(R_list)
    score_true_list = np.array(score_true_list)
    score_pred_list = np.array(score_pred_list)
    
    
    
    for R, score_true, score_pred in zip(R_list[argsort], 
                                         score_true_list[argsort],
                                         score_pred_list[argsort],
                                        ):
        
        
        [speedup_mask,
         speedup_metric,
         speedup_R] = _one_speedup_iteration(score_true,
                                             score_pred,
                                             threshold,
                                             R,
                                             speedup_mask,
                                             speedup_R,
                                             speedup_metric,
                                             metric_type,
                                            )
        
                
    return (speedup_mask, speedup_metric, speedup_R)


def _one_speedup_iteration(metric_true, 
                               metric_prediction, 
                               threshold, 
                               R,
                               speedup_mask, 
                               speedup_R, 
                               speedup_metric,
                               metric_type = 'max_best', # 'max_best' or 'min_best'
                              ):
    
        assert metric_type == 'max_best' or metric_type == 'min_best', 'metric_type can only be "max_best" or "min_best"'
        
        if metric_type == 'max_best':
            speedup_mask_tmp = metric_prediction>threshold
        elif metric_type == 'min_best':
            speedup_mask_tmp = metric_prediction<threshold
                
        speedup_mask_tmp = speedup_mask_tmp ^ speedup_mask

        speedup_mask = speedup_mask | speedup_mask_tmp

        speedup_metric = speedup_metric + speedup_mask_tmp * metric_true

        speedup_R = speedup_R + speedup_mask_tmp * R

        return [speedup_mask, speedup_metric, speedup_R]


def fit_regressor(
    x,
    y,
    regressor = None,
    plot = False,
    ax = None,   
):
    
    if regressor is None:
        from sklearn.linear_model import TheilSenRegressor
        regressor = TheilSenRegressor()
    else: 
        regressor = regressor
        
        
    regressor.fit(np.array(x)[...,np.newaxis],np.array(y))
    pred = regressor.predict(np.array(x)[...,np.newaxis])
        
    if plot == True:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y, '.')
        ax.plot(x, pred)
        ax.legend(['true', 'pred'])
        ax.title.set_text('linear self-assessment')
        ax.set_xlabel('k-space difference')
        ax.set_ylabel('image reconstruction metric')
    return regressor

def to_PSNR(metric, eps = 10e-7):
        
        PSNR = 10 * np.log10(1/(metric+eps))
        
        return PSNR