import tensorflow as tf
import numpy as np
import time
from english_provider import EnglishDataProvider
from network import Network
from utils import model_size
from word_embedding_backend import WordEmbeddingBackend

# Training
dropout       = 0.5
batch_size    = 20

# Embedding
embedding_dim = 200

# RNN
hidden_dim    = 200
num_layers    = 2

# Dataset
eigo = EnglishDataProvider()
my_x, my_y = eigo.get_word_pairs('train')

vocab_size = len(eigo.get_vocabulary())
print 'Vocabulary size:', vocab_size

train_x, train_y = eigo.get_word_pairs('train')
valid_x, valid_y = eigo.get_word_pairs('valid')
test_x , test_y  = eigo.get_word_pairs('test')

input_seq_length = len(train_x[0])
print' Sequence length:', input_seq_length

# Placeholders
input_ = tf.placeholder(tf.int32, shape=[batch_size, input_seq_length], name="input")
targets = tf.placeholder(tf.int64, [batch_size, input_seq_length], name='targets')
keep_prob = tf.placeholder(tf.float32)

# Model
embedder = WordEmbeddingBackend(vocab_size, embedding_dim)
network = Network(input_, targets, keep_prob, batch_size, vocab_size, embedding_dim, num_layers, hidden_dim, input_seq_length, embedder)

# Create session    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

# Run initializers
session.run(tf.global_variables_initializer())
session.run(embedder.reset_op)
print 'Model size:', model_size()

best_valid_loss = None
rnn_state = session.run(network.initial_state)

saver = tf.train.Saver(max_to_keep=50)

for epoch in range(25):
    epoch_start_time = time.time()
    avg_train_loss = 0.0
    count = 0
    
    for i in range(0, len(train_x) - batch_size, batch_size):
        count += 1
        start_time = time.time()

        t_loss, _, rnn_state, gradient_norm, step, _ = session.run([
            network.loss,
            network.train_op,
            network.rnn_final_state,
            network.global_norm,
            network.global_step,
            embedder.reset_op
        ], {
            input_ : train_x[i:i+batch_size],
            targets: train_y[i:i+batch_size],
            network.initial_state: rnn_state,
            keep_prob: 1.0 - dropout
        })

        avg_train_loss += 0.05 * (t_loss - avg_train_loss)

        time_elapsed = time.time() - start_time

        if count % 500 == 0:
            print '%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f s/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                    epoch, count,
                                                    len(train_x)/batch_size,
                                                    t_loss, np.exp(t_loss),
                                                    time_elapsed,
                                                    gradient_norm)

    print('Epoch training time:', time.time()-epoch_start_time)
    
    # epoch done: time to evaluate
    avg_valid_loss = 0.0
    count = 0
    rnn_state = session.run(network.initial_state)
    for i in range(0, len(valid_x) - batch_size, batch_size):
        count += 1
        start_time = time.time()

        t_loss, rnn_state = session.run([
            network.loss,
            network.rnn_final_state
        ], {
            input_ : valid_x[i:i+batch_size],
            targets: valid_y[i:i+batch_size],
            network.initial_state: rnn_state,
            keep_prob: 1.0
        })
        
        avg_valid_loss += t_loss / (len(valid_x)/batch_size)

    print "at the end of epoch:", epoch
    print "train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss))
    print "validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss))

    # learning rate update
    if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - 1.0:
        current_learning_rate = session.run(network.learning_rate)
        current_learning_rate *= 0.5
        
        if current_learning_rate < 1.e-5:
            break
        
        session.run(network.learning_rate.assign(current_learning_rate))
        print 'New LR:', current_learning_rate
    else:
        best_valid_loss = avg_valid_loss
        
        
session.close()
