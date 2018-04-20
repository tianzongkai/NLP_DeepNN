#! /usr/bin/python2.7

class Vocab:
    def __init__(self):
        self.words = open('data/vocabs.word', 'r').read().strip().split('\n')
        self.pos = open('data/vocabs.pos', 'r').read().strip().split('\n')
        self.labels = open('data/vocabs.labels', 'r').read().strip().split('\n')
        self.actions = open('data/vocabs.actions', 'r').read().strip().split('\n')

        # create a lookup word list, index of a word is from file vocabs.word
        self.word_vocab = ['word' for _ in range(len(self.words))]
        # print self.words
        for w in self.words:
            line = w.split()
            word = line[0]
            idx = int(line[1])
            self.word_vocab[idx] = word

        # create a lookup POS list, index of a pos is from file vocabs.pos
        self.pos_vocab = ['pos' for _ in range(len(self.pos))]
        for t in self.pos:
            line = t.split()
            tag = line[0]
            idx = int(line[1])
            self.pos_vocab[idx] = tag

        # create a lookup dependency labels list, index of a label is from file vocabs.labels
        self.label_vocab = ['label' for _ in range(len(self.labels))]
        for l in self.labels:
            line = l.split()
            label = line[0]
            idx = int(line[1])
            self.label_vocab[idx] = label

        # create a lookup action list, index of a action is from file vocabs.actions
        self.action_vocab = ['action' for _ in range(len(self.actions))]
        for a in self.actions:
            line = a.split()
            action = line[0]
            idx = int(line[1])
            self.action_vocab[idx] = action

    def word2id(self, word):
        return self.word_vocab.index(word) if word in self.word_vocab else self.word_vocab.index('<unk>')

    def pos2id(self, tag):
        return self.pos_vocab.index(tag) if tag in self.pos_vocab else self.pos_vocab.index('<null>')

    def label2id(self, label):
        return self.label_vocab.index(label)

    def action2id(self, action):
        return self.action_vocab.index(action)

    def id2action(self, id):
        return self.action_vocab[id]


    def num_words(self):
        return len(self.words)

    def num_pos(self):
        return len(self.pos)

    def num_labels(self):
        return len(self.labels)

    def num_actions(self):
        return len(self.actions)
