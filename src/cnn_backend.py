import tensorflow as tf
import numpy as np

class CharacterCNNBackend():
    def __init__(self, vocab_size, embedding_dim, max_word_length, kernels, kernel_features, highway_layers):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_word_length = max_word_length
        self.kernels = kernels
        self.kernel_features = kernel_features
        self.highway_layers = highway_layers
        
        # Initialize embeddings
        self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.embedding_dim])
        
        # Clear operation for 0th embedding
        self.reset_op = tf.scatter_update(self.embeddings, [0], tf.constant(0.0, shape=[1, self.embedding_dim]))
    
    def embed_word(self, inputs):
        input_embedded = tf.nn.embedding_lookup(self.embeddings, inputs)
        input_embedded = tf.reshape(input_embedded, [-1, self.max_word_length, self.embedding_dim])
        input_embedded = tf.expand_dims(input_embedded, 1)
        
        layers = []
        
        for kernel_size, kernel_feature_size in zip(self.kernels, self.kernel_features):
            reduced_length = self.max_word_length - kernel_size + 1
            
            with tf.variable_scope("kernel_%d" % kernel_size):
                w = tf.get_variable('w', [1, kernel_size, input_embedded.get_shape()[-1], kernel_feature_size])
                b = tf.get_variable('b', [kernel_feature_size])
            
            conv = tf.nn.conv2d(input_embedded, w, strides=[1, 1, 1, 1], padding='VALID') + b
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
            layers.append(tf.squeeze(pool, [1, 2]))
        
        if len(layers) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]
        
        highway_stack = [output]
        
        for i in range(self.highway_layers):
            with tf.variable_scope("highway_%d" % i):
                dim = np.sum(self.kernel_features)
                
                f_w = tf.get_variable('f_w', [dim, dim])
                f_b = tf.get_variable('f_b', [dim])
                
                t_w = tf.get_variable('t_w', [dim, dim])
                t_b = tf.get_variable('t_b', [dim])
                
                t = tf.sigmoid(tf.matmul(highway_stack[-1], t_w) + t_b)
                f = tf.matmul(highway_stack[-1], f_w) + f_b
                g = tf.nn.relu(f)
                z = t * g + (1. - t) * highway_stack[-1]
                highway_stack += [z]
        
        return highway_stack[-1]
    