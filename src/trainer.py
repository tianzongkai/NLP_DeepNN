import dynet as dy
import numpy as np
import pickle
from net_properties import *
from network import *
from vocab import *
from depModel import *
import time


TRAININT_FILE = './data/train.data' #143,758 lines

for p in [0]:
    PART = p

    if PART == 1:
        PICKLE_PATH = 'pickle_path'
        TRAINING_EPOCHS = 7
        WORD_EMBEDING_DIM = 64
        POS_EMBEDING_DIM = 32
        LABEL_EMBEDING_DIM = 32
        MINIBATCH_SIZE = 1000
        FIRST_HIDDEN_LAYER_DIM = 200
        SECOND_HIDDEN_LAYER_DIM = 200
        SAVED_MODEL_FILE_NAME = 'trained_network'
        output_p = './outputs/dev_part1.conll'

    elif PART == 2:
        PICKLE_PATH = 'pickle_path_2'
        TRAINING_EPOCHS = 7
        WORD_EMBEDING_DIM = 64
        POS_EMBEDING_DIM = 32
        LABEL_EMBEDING_DIM = 32
        MINIBATCH_SIZE = 1000
        FIRST_HIDDEN_LAYER_DIM = 400
        SECOND_HIDDEN_LAYER_DIM = 400
        SAVED_MODEL_FILE_NAME = 'trained_network_2'
        output_p = './outputs/dev_part2.conll'

    elif PART == 3:
        # activation function changed to tanh based on part 2
        TRAINING_EPOCHS = 7
        WORD_EMBEDING_DIM = 64
        POS_EMBEDING_DIM = 32
        LABEL_EMBEDING_DIM = 32
        MINIBATCH_SIZE = 1000
        FIRST_HIDDEN_LAYER_DIM = 400
        SECOND_HIDDEN_LAYER_DIM = 400
        PICKLE_PATH = 'pickle_path_3_tanh'
        SAVED_MODEL_FILE_NAME = 'trained_network_3_tanh'
        output_p = './outputs/dev_part3_tanh.conll'

    elif PART == 4:
        # !!!!!! remember to change activation function back to ReLu before training !!!!!!

        # minibatch size changed to 2000 based on part 2
        PICKLE_PATH = 'pickle_path_4_batch2k'
        TRAINING_EPOCHS = 7
        WORD_EMBEDING_DIM = 64
        POS_EMBEDING_DIM = 32
        LABEL_EMBEDING_DIM = 32
        MINIBATCH_SIZE = 2000
        FIRST_HIDDEN_LAYER_DIM = 400
        SECOND_HIDDEN_LAYER_DIM = 400
        SAVED_MODEL_FILE_NAME = 'trained_network_4_batch2k'
        output_p = './outputs/dev_part4_batch2k.conll'

    elif PART == 5:
        # training epochs changed to 11 based on part 2
        PICKLE_PATH = 'pickle_path_5_11epo'
        TRAINING_EPOCHS = 11
        WORD_EMBEDING_DIM = 64
        POS_EMBEDING_DIM = 32
        LABEL_EMBEDING_DIM = 32
        MINIBATCH_SIZE = 1000
        FIRST_HIDDEN_LAYER_DIM = 400
        SECOND_HIDDEN_LAYER_DIM = 400
        SAVED_MODEL_FILE_NAME = 'trained_network_5_11epo'
        output_p = './outputs/dev_part5_11epo.conll'

    elif PART == 6:
        # embedding layer dim changed based on part 2
        PICKLE_PATH = 'pickle_path_6_embedding'
        TRAINING_EPOCHS = 7
        WORD_EMBEDING_DIM = 100
        POS_EMBEDING_DIM = 50
        LABEL_EMBEDING_DIM = 50
        MINIBATCH_SIZE = 1000
        FIRST_HIDDEN_LAYER_DIM = 400
        SECOND_HIDDEN_LAYER_DIM = 400
        SAVED_MODEL_FILE_NAME = 'trained_network_6_embedding'
        output_p = './outputs/dev_part6_embedding.conll'

    # Create vocabularies for word, pos, dependency labels and actions
    vocab = Vocab()

    # Create object to store netword properties
    net_properties = NetProperties(WORD_EMBEDING_DIM, POS_EMBEDING_DIM, LABEL_EMBEDING_DIM, FIRST_HIDDEN_LAYER_DIM,
                                   SECOND_HIDDEN_LAYER_DIM, MINIBATCH_SIZE)

    # Save vocab and net_properties objects to a file
    pickle.dump((vocab, net_properties), open(PICKLE_PATH, 'w'))

    # Create a neural network object
    network = Network(vocab, net_properties)

    # Train the save the network
    print 'Part %r Start training...' % (PART)
    start = time.clock()
    network.train(TRAININT_FILE, TRAINING_EPOCHS)
    network.save(SAVED_MODEL_FILE_NAME)
    end = time.clock()
    print 'Part %r Training time is %2f minutes' % (PART, (end-start)/60)

    # Decode dev file
    print 'Part %r Start decoding' % (PART)
    m = DepModel(pickle_path=PICKLE_PATH, saved_network_path=SAVED_MODEL_FILE_NAME)
    Decoder(m.score, m.actions).parse('./trees/dev.conll', output_p)
    print 'Part %r Decoding done\n' % (PART)

# part 1
# print 'decode part 1...'
# m = DepModel(pickle_path='pickle_path', saved_network_path='trained_network')
# Decoder(m.score, m.actions).parse('./trees/dev.conll', './outputs/dev_part1.conll')
# Decoder(m.score, m.actions).parse('./trees/test.conll', './outputs/test_part1.conll')


# part 2
# print 'decode part 2...'
# m = DepModel(pickle_path='pickle_path_2', saved_network_path='trained_network_2')
# Decoder(m.score, m.actions).parse('./trees/dev.conll', './outputs/dev_part2.conll')
# Decoder(m.score, m.actions).parse('./trees/test.conll', './outputs/test_part2.conll')


# part 3 tanh
# print 'decode part 3 tanh'
# m = DepModel(pickle_path='pickle_path_3_tanh', saved_network_path='trained_network_3_tanh')
# Decoder(m.score, m.actions).parse('./trees/dev.conll', './outputs/dev_part3_tanh.conll')

