import tensorflow as tf

class CharacterCNNBackend():
    def __init__(self, vocab_size, embedding_dim, kernels=[1,2,3,4,5,6], kernel_features = [25,50,75,100,125,150]):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernels = kernels
        self.kernel_features = kernel_features
        
        # Initialize embeddings
        self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.embedding_dim])
        
        # Clear operation for 0th embedding
        self.reset_op = tf.scatter_update(self.embeddings, [0], tf.constant(0.0, shape=[1, self.embedding_dim]))
    
    def embed_word(self, inputs):
        max_word_length = inputs.get_shape()[1]
        inputs = tf.expand_dims(inputs, 1)
        
        layers = []
        
        for kernel_size, kernel_feature_size in zip(self.kernels, self.kernel_features):
            reduced_length = max_word_length - kernel_size + 1
            conv = tf.nn.conv2d(inputs, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
            layers.append(tf.squeeze(pool, [1, 2]))
        
        if len(layers) > 1:
            return tf.concat(layers, 1)
        else:
            return layers[0]
    