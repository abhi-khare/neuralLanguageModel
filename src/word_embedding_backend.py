import tensorflow as tf

class WordEmbeddingBackend():
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.embedding_dim])
        
        # Clear operation for 0th embedding
        self.reset_op = tf.scatter_update(self.embeddings, [0], tf.constant(0.0, shape=[1, self.embedding_dim]))
        
    def embed_word(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)
    