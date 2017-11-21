import tensorflow as tf
import numpy as np
import time, os
from data_provider import DataProvider
from network import Network
from utils import model_size
from cnn_backend import CharacterCNNBackend
from rnn_backend import CharacterRNNBackend
import sys

c2w_model = sys.argv[2]

# Training
dropout       = 0.5
batch_size    = 20

# Front end RNN
hidden_dim    = 200
num_layers    = 2

# Back end CNN
cnn_embedding_dim = 15

if sys.argv[3]=='large':
    kernels = [1,2,3,4,5,6]
else:
    kernels = [1,2,3]

kernel_features = [25*x for x in kernels]
highway_layers = int(sys.argv[4])

# Back end RNN
rnn_embedding_dim = 200

try:
    rnn_hidden_dim = int(sys.argv[3])
except:
    rnn_hidden_dim = None

rnn_output_dim = 200

# Dataset
eigo = DataProvider(sys.argv[1])

vocab_size = len(eigo.get_vocabulary())
print 'Vocabulary size:', vocab_size

train_x, _ = eigo.get_char_pairs('train')
valid_x, _ = eigo.get_char_pairs('valid')
test_x , _ = eigo.get_char_pairs('test')

_, train_y = eigo.get_word_pairs('train')
_, valid_y = eigo.get_word_pairs('valid')
_, test_y  = eigo.get_word_pairs('test')

input_seq_length = len(train_x[0])
print 'Sequence length:', input_seq_length

max_word_length = len(train_x[0][0])
print 'Maximum word length:', max_word_length
sys.stdout.flush()

# Placeholders
input_ = tf.placeholder(tf.int32, shape=[batch_size, input_seq_length, max_word_length], name="input")
targets = tf.placeholder(tf.int64, [batch_size, input_seq_length], name='targets')
keep_prob = tf.placeholder(tf.float32)

# Model
with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    if c2w_model == 'cnn':
        embedder = CharacterCNNBackend(vocab_size, cnn_embedding_dim, max_word_length, kernels, kernel_features, highway_layers)
        print kernels, kernel_features, highway_layers
    elif c2w_model == 'rnn':
        embedder = CharacterRNNBackend(vocab_size, rnn_embedding_dim, max_word_length, rnn_hidden_dim, keep_prob, rnn_output_dim, batch_size)
        print rnn_hidden_dim

    sys.stdout.flush()
    network = Network(input_, targets, keep_prob, batch_size, vocab_size, num_layers, hidden_dim, input_seq_length, embedder)

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

saver = tf.train.Saver(max_to_keep=5000)
best_filename = None

try:
    os.makedirs("saves/"+c2w_model)
except OSError:
    pass

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
            sys.stdout.flush()

    print('Epoch training time:', time.time()-epoch_start_time)
    sys.stdout.flush()

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
    sys.stdout.flush()
 
    save_filename = 'saves/%s/%s_%s_%s_epoch%03d_%.4f.model' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], epoch, avg_valid_loss)
    saver.save(session, save_filename)

    # learning rate update
    if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - 1.0:
        current_learning_rate = session.run(network.learning_rate)
        current_learning_rate *= 0.5
        
        if current_learning_rate < 1.e-5:
            break
        
        session.run(network.learning_rate.assign(current_learning_rate))
        print 'New LR:', current_learning_rate
        sys.stdout.flush()
    else:
        best_valid_loss = avg_valid_loss
        best_filename = save_filename        

# Load the best performing model
saver.restore(session, best_filename)
print "restoring saved model", best_filename
sys.stdout.flush()

# Test the model
rnn_state = session.run(network.initial_state)

count = 0
avg_test_loss = 0
start_time = time.time()

for i in range(0, len(test_x) - batch_size, batch_size):
    count += 1

    t_loss, rnn_state = session.run([
        network.loss,
        network.rnn_final_state
    ], {
        input_ : test_x[i:i+batch_size],
        targets: test_y[i:i+batch_size],
        network.initial_state: rnn_state,
        keep_prob: 1.0
    })
    
    avg_test_loss += t_loss / (len(test_x)/batch_size)

print "test loss = %6.8f, perplexity = %6.8f" % (avg_test_loss, np.exp(avg_test_loss))
print "test samples:", count * batch_size
sys.stdout.flush()

session.close()
