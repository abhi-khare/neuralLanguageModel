import tensorflow as tf
from lazy_property import LazyProperty

class Network():
    INITIAL_LR = 1.0
    MAX_GRAD_NORM = 5.0
    
    def __init__(self, inputs, targets, keep_prob, vocab_size, embedding_dim, num_layers, hidden_dim, sequence_length, embedder):
        # Set the placeholders
        self.inputs = inputs
        self.targets = targets
        self.keep_prob = keep_prob
        
        # Embedder network
        self.embedder = embedder
        
        # Set the basic parameters
        self.vocab_size = vocab_size
        self.input_sequence_length = sequence_length
        
        # Set the hyperparameters
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    @LazyProperty
    def logits(self):
        # LSTM
        if self.num_layers == 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, state_is_tuple=True, forget_bias=0.0),
                    output_keep_prob=self.keep_prob
                   )
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([
                        tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, state_is_tuple=True, forget_bias=0.0),
                            output_keep_prob=self.keep_prob
                        )
                        for _ in range(self.num_layers)
                    ])
        
        self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        
        # Invoke embed operation
        input_embedded = self.embedder.embed(self.inputs)
        
        # Set up inputs to the LSTM
        rnn_inputs = tf.reshape(input_embedded, [self.batch_size, self.input_seq_length, -1])
        rnn_inputs = [tf.squeeze(x, [1]) for x in tf.split(rnn_inputs, self.input_seq_length, 1)]
        
        # Retrieve the result
        self.rnn_outputs, self.rnn_final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, self.initial_state)
        
        # Output Projection
        output_W = tf.get_variable('outW', shape=[self.hidden_dim, self.vocab_size])
        output_b = tf.get_variable('outB', shape=[self.vocab_size])
        
        logits = []
        
        for output in self.rnn_outputs:
            logits += [ tf.matmul(output, output_W) + output_b ]
            
        return logits    
    
    @LazyProperty
    def loss(self):
        labels = [tf.squeeze(x, [1]) for x in tf.split(self.targets, self.input_seq_length, 1)]        
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=labels))
    
    @LazyProperty
    def train_op(self):        
        # SGD learning parameter
        self.learning_rate = tf.Variable(Network.INITIAL_LR, trainable=False, name='learning_rate')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
            
        # collect all trainable variables and clip gradients
        tvars = tf.trainable_variables()
        grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.loss * self.input_seq_length, tvars), Network.MAX_GRAD_NORM)
    
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)