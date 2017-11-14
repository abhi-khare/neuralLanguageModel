import tensorflow as tf
import sys

class CharacterRNNBackend():
    def __init__(self, vocab_size, embedding_dim, max_word_length, hidden_dim, keep_prob, output_dim, batch_size):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_word_length = max_word_length
        self.hidden_dim = hidden_dim
        self.keep_prob = keep_prob
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # Initialize embeddings
        self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.embedding_dim])
        
        # Clear operation for 0th embedding
        self.reset_op = tf.scatter_update(self.embeddings, [0], tf.constant(0.0, shape=[1, self.embedding_dim]))
    
    # Based on [LSTM Language Model with Subword Units Input Representations](https://github.com/claravania/subword-lstm-lm)
    def embed_word(self, inputs):
        input_embedded = tf.nn.embedding_lookup(self.embeddings, inputs)
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        
        with tf.variable_scope("c2w"):
            with tf.variable_scope("forward"):
                self._fw_cell = cell_fn(self.hidden_dim, forget_bias=0.0)
                self._fw_cell = tf.nn.rnn_cell.DropoutWrapper(self._fw_cell, self.keep_prob)
            
            with tf.variable_scope("backward"):
                self._bw_cell = cell_fn(self.hidden_dim, forget_bias=0.0)
                self._bw_cell = tf.nn.rnn_cell.DropoutWrapper(self._bw_cell, self.keep_prob)
                
            rnn_inputs = tf.nn.dropout(input_embedded, self.keep_prob)
            rnn_inputs = tf.split(rnn_inputs, self.batch_size, 0)
            rnn_inputs = [tf.squeeze(input_, [0]) for input_ in rnn_inputs]
            
            c2w_outputs = []
            softmax_w_fw = tf.get_variable("softmax_fw", [self.hidden_dim * 2, self.output_dim])
            softmax_w_bw = tf.get_variable("softmax_bw", [self.hidden_dim * 2, self.output_dim])
            b_c2w = tf.get_variable("c2w_biases", [self.output_dim])
            
            for i in range(len(rnn_inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                
                _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(self._fw_cell, self._bw_cell, rnn_inputs[i], dtype=tf.float32)
                
                fw_param = tf.matmul(tf.concat(fw_state, 1), softmax_w_fw)
                bw_param = tf.matmul(tf.concat(bw_state, 1), softmax_w_bw)
                final_output = fw_param + bw_param + b_c2w
                c2w_outputs.append(tf.expand_dims(final_output, 0))
                
            c2w_outputs = tf.concat(c2w_outputs, 0)
            
        return c2w_outputs
        
    