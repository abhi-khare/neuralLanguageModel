import tensorflow as tf

class CharacterRNNBackend():
    def __init__(self, vocab_size, embedding_dim, kernels=[1,2,3,4,5,6], kernel_features = [25,50,75,100,125,150]):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernels = kernels
        self.kernel_features = kernel_features
        
        # Initialize embeddings
        self.embeddings = tf.get_variable('py', [self.vocab_size, self.embedding_dim])
        
        # Clear operation for 0th embedding
        self.reset_op = tf.scatter_update(self.embeddings, [0], tf.constant(0.0, shape=[1, self.embedding_dim]))
    
    def embed_word(self, inputs):
        pass
    