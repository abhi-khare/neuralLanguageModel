import numpy as np

class DataProvider:
    def __init__(self, language):
        self.tokens = {}

        for fold in ['train', 'valid', 'test']:
            self.tokens[fold] = []

            with open('../data/' + language + '/' + fold + '.txt') as f:
                for line in f:
                    line = line.strip()
                    line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ').replace('+', '')
                    
                    for word in line.split():
                        self.tokens[fold].append(word)

                    self.tokens[fold].append('+')
            
        self.word_vocab  = ['|']
        self.char_vocab  = []
        self.max_wordlen = 0

        for fold in ['train', 'valid', 'test']:
            for word in self.tokens[fold]:
                if word not in self.word_vocab:
                    self.word_vocab += [word]
                    
                    if len(word) > self.max_wordlen:
                        self.max_wordlen = len(word)
                    
                    for character in list(word.decode('utf8')):
                        if character not in self.char_vocab:
                            self.char_vocab += [character]
        
        # sort the alphabet
        self.char_vocab = [' ', '{', '}', '|', '+'] + sorted(self.char_vocab)
        
        self.reverse_dict = dict([(self.word_vocab[i], i) for i in range(len(self.word_vocab))])
        self.reverse_alph = dict([(self.char_vocab[i], i) for i in range(len(self.char_vocab))])
        
    def get_vocabulary(self):
        return self.word_vocab
    
    def get_alphabet(self):
        return self.char_vocab
    
    def get_word_pairs(self, fold, seq_len=35, batch_size=20):
        def lookup_word(x):
            return self.reverse_dict[x]
    
        x = np.array([map(lookup_word, self.tokens[fold][i:i+seq_len]) for i in range(0, len(self.tokens[fold]) - 1, seq_len)][:-1])
        y = np.array([map(lookup_word, self.tokens[fold][i:i+seq_len]) for i in range(1, len(self.tokens[fold]), seq_len)][:-1])
        
        x = x[:len(x)/batch_size*batch_size,:].reshape([batch_size, -1, seq_len])
        x = x.transpose([1,0,2]).reshape([-1, seq_len])
        
        y = y[:len(y)/batch_size*batch_size,:].reshape([batch_size, -1, seq_len])
        y = y.transpose([1,0,2]).reshape([-1, seq_len])
        
        return x, y
    
    def get_char_pairs(self, fold, seq_len=35, batch_size=20):
        def lookup_word(x):
            this_word = np.zeros(self.max_wordlen + 2, np.int32)
            this_word[:len(x) + 2] = np.array([self.reverse_alph[y] for y in '{' + x.decode('utf8') + '}'])
            return this_word
    
        x = np.array([map(lookup_word, self.tokens[fold][i:i+seq_len]) for i in range(0, len(self.tokens[fold]) - 1, seq_len)][:-1])
        y = np.array([map(lookup_word, self.tokens[fold][i:i+seq_len]) for i in range(1, len(self.tokens[fold]), seq_len)][:-1])
        
        x = x[:len(x)/batch_size*batch_size,:].reshape([batch_size, -1, seq_len, self.max_wordlen+2])
        x = x.transpose([1,0,2,3]).reshape([-1, seq_len, self.max_wordlen+2])
        
        y = y[:len(y)/batch_size*batch_size,:].reshape([batch_size, -1, seq_len, self.max_wordlen+2])
        y = y.transpose([1,0,2,3]).reshape([-1, seq_len, self.max_wordlen+2])
        
        return x, y
