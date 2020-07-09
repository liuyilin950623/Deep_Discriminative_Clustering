## Discriminative layer and later Clustering layer

import keras.backend as K
import tensorflow as tf
import keras.layers as Layers
from keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()

class Discriminative(Layers.Layer):
    
    def __init__(self, alpha, batch_size, k, **kwargs):
        super(Discriminative, self).__init__(**kwargs)
        self.alpha = alpha
        self.batch_size = batch_size
        self.k = k
              
    def call(self, inputs):
        original, hidden = inputs
        
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
        anchor_matrix = tf.cast(ranks < self.k, dtype = tf.float32)
        
        ### Symmterise Anchor Matrix
        anchor_matrix = anchor_matrix + tf.transpose(anchor_matrix)
        anchor_matrix = tf.minimum(anchor_matrix, tf.ones_like(anchor_matrix))
        
        matrix_of_1s = tf.ones([self.batch_size, self.batch_size], dtype = tf.float32)
        non_anchor_matrix = matrix_of_1s - anchor_matrix
        
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
        similarity = tf.nn.l2_normalize(hidden, axis = 1)
        C = tf.matmul(similarity, tf.transpose(similarity))     
        
        ## Calculate Weights later to weigh anchor points and non-anchor points
        Nw = (1 - self.alpha)/tf.reduce_sum(anchor_matrix)
        Nb = 1/(tf.reduce_sum(non_anchor_matrix) - tf.reduce_sum(anchor_matrix))
        
        ## Depends on Anchor Pairs of not, maximise the L1 sum of anchor matrix
        non_anchor = Nb * tf.multiply(C, non_anchor_matrix)
        anchor = Nw * tf.multiply(C, anchor_matrix)
        
        return [non_anchor, anchor]