"""
    For more details, please read/cite the article:
    
    ...
    
    For the original version of LOUPE, please read:
    
    Bahadir, C.D., Wang, A.Q., Dalca, A.V., Sabuncu, M.R. 
    "Deep-Learning-Based Optimization of the Under-Sampling Pattern in MRI"
    (2020) IEEE Transactions on Computational Imaging,
    6, art. no. 9133281, pp. 1139-1152. 
    DOI:10.1109/TCI.2020.3006727.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.initializers import Initializer
import numpy as np
from tensorflow import convert_to_tensor

class RescaleProbMap(Layer):
    """
        Rescale Probability Mask

        given a prob map x, rescales it to get the desired speed-up "R" 

        (r=1/R)

        if mean(x) >= 1/R: x' = x*r/mean(x)
        if mean(x) < 1/R:  x' = 1 - (1-x)*(1-r)/(1-mean(x))
    """
    
    def __init__(self, R, **kwargs):
        self.R = R
        super(RescaleProbMap, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({'R':self.R})
        return config

    def build(self, input_shape):
        super(RescaleProbMap, self).build(input_shape)

    def call(self, x):
        xbar = K.mean(x)
        r = 1/(self.R * xbar)
        beta = (1-1/self.R) / (1-xbar)
        
        # compute adjucement
        le = tf.cast(tf.less_equal(r, 1), tf.float32)   
        return  le * x * r + (1-le) * (1 - (1 - x) * beta)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def set_attributes(self, R = None,):
        if R != None:
            self.R = R
        return
    
    
class ProbMask(Layer):
    """ 
        Probability mask layer

        Contains a layer of weights, that is then passed through a sigmoid.
        The sigmoid is controlled by a sigma "t" multiplication factor 
        (non-critical hyper-parameter).

        The default initialization returns a uniform distributed mask 
        after teh sigmoid is applied.

        The mask can be read or written by the user, using
        ".read_probMask" ".write_probMask"
    """
    
    def __init__(self, 
                 slope = 10,
                 initializer = None,
                 trainable = True,
                 cartesian_type = None,
                 **kwargs):
        
        assert (cartesian_type is None or
                cartesian_type == 'sense_columns' or 
                cartesian_type == 'sense_rows'), 'only None, "sense_columns"\
                or "sense_rows" can be given to "cartesian_type".'
        
        with tf.init_scope():
            if initializer == None:
                self.initializer = self._logit_slope_random_uniform
            else:
                self.initializer = initializer

        self.cartesian_type = cartesian_type
        self.slope = tf.Variable(slope, dtype=tf.float32)
        self.trainable = trainable
        super(ProbMask, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'slope':self.slope,
            'initializer':self.initializer,
            'trainable':self.trainable,
            'cartesian_type':self.cartesian_type,
        })
        return config

    def build(self, input_shape):
        """
            takes as input the input data, which is [N x ... x 2] 
        """
        lst = list(input_shape)
        lst[-1] = 1
        input_shape = tuple(lst)
        
        if self.cartesian_type is None:
            input_shape_gamma = input_shape[1:]
            self.cartesian_handle = self.do_nothing
            
        elif self.cartesian_type == 'sense_rows':
            input_shape_gamma = (input_shape[1], 1)
            self.cartesian_handle = self.sense_rows
        
        elif self.cartesian_type == 'sense_columns':
            input_shape_gamma = (input_shape[2], 1)
            self.cartesian_handle = self.sense_columns
        
        with tf.init_scope():
            self.gamma = self.add_weight(name='logit_weights', 
                                         shape=input_shape_gamma,
                                         initializer=self.initializer,
                                         trainable=self.trainable,
                                        )
        
        super(ProbMask, self).build(input_shape)
        
        
    
    def sense_columns(self, x):
        return tf.expand_dims(tf.linalg.matmul(tf.ones(tf.shape(x)), x, transpose_b = True), -1)
    
    def sense_rows(self, x):
        return tf.expand_dims(tf.linalg.matmul(x, tf.ones(tf.shape(x)), transpose_b = True), -1)
    
    def do_nothing(self, x):
        return x
    
    
    
    def call(self,x):
        """
            "gamma" is multiplied with the zeroed entry (0*x[..., 0:1])
            so the output inherits the "batch size" dimension
            "gamma" is then multiplied with the slope "s" and passed 
            through the sigmoid to create the probability mask
        """
        
        logit_weights = 0*x[..., 0:1] + self.cartesian_handle(self.gamma)
        prob_mask = tf.sigmoid(self.slope * logit_weights)
        return prob_mask

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[-1] = 1
        return tuple(lst)

    def read_probMask(self, return_gamma = False, ):
        """
            if apply_sigmoid == True, then the probMask is returned
            else, the matrix "gamma" controlling the probMask is returned
        """
        
        gamma = self.gamma
        if return_gamma == True:
            return gamma
        else:
            prob_mask = tf.sigmoid(self.slope * gamma)
            return self.cartesian_handle(prob_mask)
        
    def write_probMask(self, probMask, revert_sigmoid = True):
        """
            if revert_sigmoid == True, then "gamma" is updated by 
            first applying the logit to the probMask
            else, the matrix "gamma" is directly updated 
            (it is assumed that "gamma" is directly given)
        """
        if revert_sigmoid==True:
            probMask = - tf.math.log(1. / probMask - 1.) / self.slope
        
        self.gamma = tf.Variable(probMask,
                                 name='logit_weights',
                                 trainable=self.trainable,
                                )
        return self.gamma
    
    def _logit_slope_random_uniform(self, shape, dtype=None, eps=0.01):
        x = tf.random.uniform(shape, minval = eps, maxval = 1.0 - eps)
        return - tf.math.log(1. / x - 1.) / self.slope
    
    
    def set_attributes(self, slope = None, trainable = None, ):
        if slope != None:
            self.slope = slope
        if trainable != None:
            self.trainable = trainable
        return
    
    
class FixedMask(Layer):
    """ 
        Probability mask layer

        Contains a layer of weights, that is then passed through a sigmoid.
        The sigmoid is controlled by a sigma "t" multiplication factor 
        (non-critical hyper-parameter).

        The default initialization returns a uniform distributed mask 
        after teh sigmoid is applied.

        The mask can be read or written by the user, using
        ".read_probMask" ".write_probMask"
    """
    
    def __init__(self, 
                 mask,
                 **kwargs):
        
        self.mask = tf.cast(mask, tf.float32)
        super(FixedMask, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mask':self.mask,
        })
        return config

    def build(self, input_shape):
        
        super(FixedMask, self).build(input_shape)
    
    
    def call(self,x):
        """
            "gamma" is multiplied with the zeroed entry (0*x[..., 0:1])
            so the output inherits the "batch size" dimension
            "gamma" is then multiplied with the slope "s" and passed 
            through the sigmoid to create the probability mask
        """
        
        mask = 0*x[..., 0:1] + self.mask
        return mask

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[-1] = 1
        return tuple(lst)
    
    
class ThresholdRandomMask(Layer):
    """ 
        Threshold Probability mask layer
        
        
        Takes two inputs having the same shape.
        The output has the same shape of the first/second input, 
        each element is the approximation of a Bernoullian distributed value.
        Each elemennt of the first entry gives the probability 
        p of the correspondent output
        Each element of the second is used as a comparison
        , e.g., if the element is drawn from a random uniform distribution,
        each element of the ouput is Bernoullian with 
        distribution p when output = p > random uniform number
        Instead of using the ">" operation a sigmoid is used 
        (if ">" was used the gradient would be constant 0)
    """

    def __init__(self, slope = 200, **kwargs):
        
        self.slope = tf.Variable(initial_value = slope,
                                 trainable = False,
                                 name = 'thresh_slope',
                                 dtype=tf.float32) 
        
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'slope':self.slope})
        return config

    def build(self, input_shape):
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self, x):
        inputs = x[0]
        thresh = x[1]
        return tf.sigmoid(self.slope * (inputs-thresh))

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def set_attributes(self, slope = None, trainable = None, ):
        if slope != None:
            self.slope = tf.Variable(initial_value = slope,
                                     trainable = False,
                                     name = 'thresh_slope',
                                     dtype=tf.float32,
                                    ) 
        if trainable != None:
            self.trainable = trainable
        return
    
class RandomMask(Layer):
    """ 
        Random Uniform Matrix for comparison to the ProbMask 
        
        Create a random mask of the same size as the input shape
        
        maxval and minval can be modified by the user with the methods 
        ".set_maxmin()".
    """

    def __init__(self, minval = 0.0, maxval = 1.0, cartesian_type = None, **kwargs):
        self.maxval = maxval
        self.minval = minval
        
        assert (cartesian_type is None or
                cartesian_type == 'sense_columns' or 
                cartesian_type == 'sense_rows'), 'only None, "sense_columns"\
                or "sense_rows" can be given to "cartesian_type".'
        
        self.cartesian_type = cartesian_type
        super(RandomMask, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxval':self.maxval,
            'minval':self.minval,
            'cartesian_type':self.cartesian_type,
        })
        return config

    def build(self, input_shape):
        lst = list(input_shape)
        lst[-1] = 1
        input_shape = tuple(lst)
        
        if self.cartesian_type is None:
            self.shape_thresh_handle = self.do_nothing
            self.cartesian_handle = self.do_nothing
            
        elif self.cartesian_type == 'sense_rows':
            self.shape_thresh_handle = self.sense_rows_shape
            self.cartesian_handle = self.sense_rows
        
        elif self.cartesian_type == 'sense_columns':
            self.shape_thresh_handle = self.sense_columns_shape
            self.cartesian_handle = self.sense_columns
        
        
        super(RandomMask, self).build(input_shape)
    
    
    
    def do_nothing(self, x):
        return x
    
    def sense_columns(self, x):
        return tf.expand_dims(
            tf.linalg.matmul(tf.ones(tf.shape(x)), 
                             x,
                             transpose_b = True),
            -1,
        )
    
    def sense_rows(self, x):
        return tf.expand_dims(
            tf.linalg.matmul(x, 
                             tf.ones(tf.shape(x)),
                             transpose_b = True),
            -1,
        )
    
    def sense_rows_shape(self, x):
        return (x[0], ) + (1, )
    
    def sense_columns_shape(self, x):
        return (x[1], ) + (1, )
    
    
    
    def call(self,x):
        
        threshs = K.random_uniform(self.shape_thresh_handle(x.shape[1:]), 
                                   minval=self.minval,
                                   maxval=self.maxval,
                                   dtype='float32',
                                  )
        
        return self.cartesian_handle(threshs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def set_attributes(self, maxval = None, minval = None):
        if maxval != None:
            self.maxval = maxval
        if minval != None:
            self.minval = minval
        return
    
    
class Masker(Layer):
    
    def __init__(self, masker = None, **kwargs):
        self.masker = masker
        super(Masker, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(Masker,self).get_config().copy()
        config.update({'masker':self.masker,
                      })
        return config
    
    def build(self, input_shape):
        if self.masker is None:
            self.masker = tf.ones(input_shape[1:])
            
        super(Masker, self).build(input_shape)
    
    def call(self, inputs):
        return tf.math.multiply(inputs, self.masker)
    
    def write_masker(self, masker):
        self.masker = masker
        return self.maskingMask
    
    def read_masker(self, masker):
        return self.masker
        
    def compute_output_shape(self, input_shape):
        return input_shape

    
    

class ComplexAbs(Layer):
    """
        Absolute Value of Complex Numbers
    """

    def __init__(self, **kwargs):
        super(ComplexAbs, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ComplexAbs, self).build(input_shape)

    def call(self, inputs):
        two_channel = tf.complex(inputs[..., 0], inputs[..., 1])
        two_channel = tf.expand_dims(two_channel, -1)
        
        two_channel = tf.abs(two_channel)
        two_channel = tf.cast(two_channel, tf.float32)
        return two_channel

    def compute_output_shape(self, input_shape):
        list_input_shape = list(input_shape)
        list_input_shape[-1] = 1
        return tuple(list_input_shape)
    
class UnderSampleHolistic(Layer):
    """
        Undersampling by multiplication of k-space with the mask

        Inputs: [kspace (2-channel), mask (single-channel)]
        
        if hard_threshold == True, a threshold (>0.5) is applied
        after the elementwise multiplication to every element
        consider to set it ONLY for inference evaluations
        
        if complement == True, the complement of the mask is used 
        instead of the mask, e.g., m'= 1-m (m values are in [0,1])
        In other words, assuming mask "m" is binary, 
        if complement == False, the element of the k-space where m == 0
        are zeroed out,
        vice verse if complement == True, 
        the elements of the k-space where m == 1 are zeroed out.
    """

    def __init__(self, hard_threshold = False, complement = False, **kwargs):
        self.hard_threshold = hard_threshold
        self.complement = complement
        super(UnderSampleHolistic, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(UnderSampleHolistic,self).get_config().copy()
        config.update({'hard_threshold':self.hard_threshold,
                       'complement':self.complement,
                      })
        return config

    def build(self, input_shape):
        super(UnderSampleHolistic, self).build(input_shape)

    def call(self, inputs):
        complement = tf.cast(self.complement, tf.float32)
        hard_threshold = tf.cast(self.hard_threshold, tf.float32)
        
        mask = (1-complement) * inputs[1][...,0] + complement * (1-inputs[1][...,0])
        mask = (1-hard_threshold) * mask + hard_threshold * tf.cast(
                    tf.keras.backend.greater(mask, 0.5),tf.float32)
        
        k_space_r = tf.multiply(inputs[0][..., 0], mask)
        k_space_i = tf.multiply(inputs[0][..., 1], mask)

        k_space = tf.stack([k_space_r, k_space_i], axis = -1)
        k_space = tf.cast(k_space, tf.float32)
        return k_space

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def set_attributes(self, complement = None, hard_threshold = None):
        if complement != None:
            self.complement = complement
        if hard_threshold != None:
            self.hard_threshold = hard_threshold
        return
    
class ConcatenateZero(Layer):
    """
    Concatenate input with a zero'ed version of itself

    Input: tf.float32 of size [batch_size, ..., n]
    Output: tf.float32 of size [batch_size, ..., n*2]
    """

    def __init__(self, **kwargs):
        super(ConcatenateZero, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatenateZero, self).build(input_shape)

    def call(self, inputx):
        return tf.concat([inputx, inputx*0], -1)


    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[-1] *= 2
        return tuple(input_shape_list)
    
    
class Clip(Layer):
    
    def __init__(self, **kwargs):
        super(Clip, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Clip, self).build(input_shape)

    def call(self, x):
        clipped = tf.clip_by_value(x, 0, 1)
        return clipped

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class FFT(Layer):
    """
    fft layer, assuming the real/imag are input/output via two features

    Input: tf.float32 of size [batch_size, ..., 2]
    Output: tf.float32 of size [batch_size, ..., 2]
    """
    
    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        assert input_shape[-1] == 2, 'input has to have two features'
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1,2,3], 'only 1D, 2D or 3D supported'

        # super
        super(FFT, self).build(input_shape)

    def call(self, inputx):
        assert inputx.shape.as_list()[-1] == 2, 'input has to have two features'

        # get the right fft
        if self.ndims == 1:
            fft = tf.signal.fft
        elif self.ndims == 2:
            fft = tf.signal.fft2d
        else:
            fft = tf.signal.fft3d

        # get fft complex image
        fft_im = fft(tf.complex(inputx[..., 0], inputx[..., 1]))

        # go back to two-feature representation
        fft_im = tf.stack([tf.math.real(fft_im), tf.math.imag(fft_im)],
                          axis=-1)
        return tf.cast(fft_im, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class IFFT(Layer):
    """
        ifft layer

        Input: tf.float32 of size [batch_size, ..., 2]
        Output: tf.float32 of size [batch_size, ..., 2]
    """

    def __init__(self, **kwargs):
        super(IFFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        assert input_shape[-1] == 2, 'input has to have two features'
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1,2,3], 'only 1D, 2D or 3D supported'

        # super
        super(IFFT, self).build(input_shape)

    def call(self, inputx):
        assert inputx.shape.as_list()[-1] == 2, 'input has to have two features'

        # get the right fft
        if self.ndims == 1:
            ifft = tf.signal.ifft
        elif self.ndims == 2:
            ifft = tf.signal.ifft2d
        else:
            ifft = tf.signal.ifft3d

        # get ifft complex image
        ifft_im = ifft(tf.complex(inputx[..., 0], inputx[..., 1]))

        # go back to two-feature representation
        ifft_im = tf.stack([tf.math.real(ifft_im), 
                            tf.math.imag(ifft_im)], axis=-1)
        return tf.cast(ifft_im, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    
############# Self-Assessment #############

class PolynomialPower(Layer):
    """
    For every input returns an output that is the input elevated to all the power equal minor than q

    Used to create the inpubt for a Dense that learns the best polynomial coefficients
    
    given  input  = [x]      and    q>=1
    return output = [x^1, x^2, ..., x^q]
    """
    def __init__(self, q, **kwargs):
        self.q = q
        super(PolynomialPower, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'q':self.q})
        return config

    def build(self, input_shape):
        super(PolynomialPower, self).build(input_shape)

    def call(self, x):

        last_tensor = tf.repeat(x, self.q, axis=-1)
        
        exponentials = tf.range(1,self.q+1)

        last_tensor = tf.pow(tf.cast(last_tensor, tf.float32), tf.cast(exponentials, tf.float32))
        
        return  last_tensor

    
    def compute_output_shape(self, input_shape):
        
        lst = list(output_shape)
        lst[-1] = q
        output_shape = tuple(lst)
        
        print(output_shape)
        
        return output_shape
    

    
class _InitializerLinRegDense(Initializer):
    
    # Gives the passed values as initial values and pads with zeros where values miss
  
    def __init__(self, init_values, **kwargs):
        self.init_values = init_values
        
        super(_InitializerLinRegDense, self).__init__(**kwargs)
        
    def __call__(self, shape, dtype=None):
        
        init_values = np.zeros(shape)
        for i, val in enumerate(self.init_values):
            init_values[i] = val
        
        return init_values
    
    def get_config(self):  # To support serialization
        return {'init_values': self.init_values}