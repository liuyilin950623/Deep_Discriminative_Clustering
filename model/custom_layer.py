## Discriminative layer and later Clustering layer

import keras.backend as K
import tensorflow as tf
import keras.layers as Layers
from keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()

class Discriminative(Layers.Layer):
    
    def __init__(self, alpha, **kwargs):
        super(Discriminative, self).__init__(**kwargs)
        self.alpha = alpha
              
    def call(self, inputs):
        original, hidden = inputs
        self.batch_size = hidden.shape[1]
        self.k = self.batch_size // 10
        
        ######################## Find Anchor Pairs ##################################
        """ Computes pairwise L2 distances between each elements in the original input.
            Args:
                A, [m, d] matrix on important features as a tensor object
            Returns:
                distance_matrix, [m,m] matrix of pairwise distances as a tensor object
            """
        p = tf.expand_dims(original, 1) ### Expand on the columns
        q = tf.expand_dims(original, 0) ### Expand on the rows
        distance_matrix = K.sqrt(tf.reduce_sum(tf.math.squared_difference(p, q), 2))
          
        """ Computes anchor pairs and non-anchor paris based on distance matrix.
            Args:
                distance_matrix, [m,m] matrix of pairwise distances as a tensor object
            """
        idx = tf.argsort(distance_matrix, direction = 'ASCENDING') ## sort elements
        ranks = tf.argsort(idx, direction = 'ASCENDING') ## ranks
        anchor_matrix = tf.cast(ranks <= self.k, dtype = tf.float32)
        non_anchor_matrix = tf.cast(ranks > self.k, dtype = tf.float32)
        
        ######################## Compute Weighted L1 Norm #################################
        """Computes anchor pairs and non-anchor pairs based on distance matrix.
            Args:
                Latent_space Projection, tensor object
                alpha, regularisation coefficient
                anchor_matrix, 1 for anchor pairs and 0 for non-anchor pairs, tensor object
                non_anchor_matrix, 1 for non-anchor pairs and 0 for anchor pairs, tensor object
            Returns:
                Lantent Loss, Weighted L1 Norm of the Consine Distance Matrix
            """ 
        ## Calculate Cosine Similarity Matrix C
        similarity = tf.nn.l2_normalize(hidden, axis = 1) ## double check the normalisation
        C = tf.matmul(similarity, tf.transpose(similarity))
        C_abs = K.abs(C)
        
        ## Calculate Weights later to weigh anchor points and non-anchor points
        self.Nw = (1 - self.alpha)/(self.batch_size * self.k)
        self.Nb = 1/(self.batch_size** 2 - (self.batch_size * self.k))
        
        ## Depends on Anchor Pairs of not
        non_anchor = self.Nb * tf.multiply(C_abs, non_anchor_matrix)
        anchor = self.Nw * tf.multiply(C, anchor_matrix)
        
        return [non_anchor, anchor]