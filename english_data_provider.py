import numpy as np

class EnglishDataProvider:
    def __init__(self):
        self.tokens = {}

        for fold in ['train', 'valid', 'test']:
            self.tokens[fold] = []

            with open('data/' + fold + '.txt') as f:
                for line in f:
                    for word in line.split():
                        if word == '<unk>':
                            self.tokens[fold].append('|')
                        else:
                            self.tokens[fold].append(word)

                    self.tokens[fold].append('+')
            
        self.vocab = ['|']

        for fold in ['train', 'valid', 'test']:
            for word in self.tokens[fold]:
                if word not in self.vocab:
                    self.vocab += [word]
        
        self.reverse_dict = dict([(self.vocab[i], i) for i in range(len(self.vocab))])
        
    def get_vocabulary(self):
        return self.vocab
    
    def get_word_pairs(self, fold, seq_len=35):
        x = np.array([map(lambda x: self.reverse_dict[x], self.tokens[fold][i:i+seq_len]) for i in range(0, len(self.tokens[fold]) - 1, seq_len)][:-1])
        y = np.array([map(lambda x: self.reverse_dict[x], self.tokens[fold][i:i+seq_len]) for i in range(1, len(self.tokens[fold]), seq_len)][:-1])
        
        return x, y