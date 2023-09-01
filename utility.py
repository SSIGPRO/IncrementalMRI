import numpy as np
import os
import h5py
from tensorflow import config

################ General Utilities ################

def load_dataset(dataset='PD'):
    
    if dataset=='PD' or dataset=='PDFS':
        trainDataPath = os.path.join('..','datasets','kneeMRI','fastMRI_trainSet_esc_320_'+dataset+'.hdf5') 
        valDataPath = os.path.join('..','datasets','kneeMRI','fastMRI_valSet_esc_320_'+dataset+'.hdf5')
        testDataPath = os.path.join('..','datasets','kneeMRI','fastMRI_testSet_esc_320_'+dataset+'.hdf5')

        xdata = loadData(trainDataPath) # images are already rescaled: 1 is the max value of the whole volume
        vdata = loadData(valDataPath) # already split validation set
        tdata = loadData(testDataPath)
        

    elif dataset=='IXI':
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


def loadData(data_path): # load from the given folder the h5py dataset
    with h5py.File(data_path, 'r') as f:
        xdata = f[os.path.join('dataset')] [()]
    return xdata

def handle_GPUs(GPUs=None, enable_GPU=1):
    """
        For the specified GPUs, "memory growth" is activated.
        The un-specified GPUs are turned invisible to the kernel.
        
        GPUs is a string contraining the number of GPUs (e.g., GPUs = '0,1')
        If "GPUs==None" all GPUs are activated.
    """
    
    if GPUs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs
    
    physical_gpus = config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for _gpu in physical_gpus:
                config.experimental.set_memory_growth(_gpu, enable_GPU)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    logical_gpus = config.list_logical_devices('GPU')
    print('number of Logical GPUs:', len(logical_gpus))
    pass