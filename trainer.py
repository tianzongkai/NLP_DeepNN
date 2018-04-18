import dynet as dy
import numpy as np
from net_properties import *
from network import *
from vocab import *
import time

TRAININT_FILE = 'data/train.data'
TRAINING_EPOCHS = 1
WORD_EMBEDING_DIM = 64
POS_EMBEDING_DIM = 32
LABEL_EMBEDING_DIM = 32
FIRST_HIDDEN_LAYER_DIM = 200
SECOND_HIDDEN_LAYER_DIM = 200
MINIBATCH_SIZE = 1000

# Create vocabularies for word, pos, dependency labels and actions
vocab = Vocab()

# Create object to store netword properties
net_properties = NetProperties(WORD_EMBEDING_DIM, POS_EMBEDING_DIM, LABEL_EMBEDING_DIM, FIRST_HIDDEN_LAYER_DIM,
                               SECOND_HIDDEN_LAYER_DIM, MINIBATCH_SIZE)

# Create a neural network object
network = Network(vocab, net_properties)

start = time.clock()

network.train(TRAININT_FILE, TRAINING_EPOCHS)

end = time.clock()
print 'Training time is %2f s' % (end-start)