import tensorflow as tf
import numpy as np
import sys


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers,
            learning_rate=1,
            use_lstm=False,
            model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.model_name = model_name

        # placeholders
        tf.reset_default_graph()

        with tf.name_scope("input"):
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [tf.placeholder(shape=[None,],
                                          dtype=tf.int64,
                                          name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
            self.labels = [tf.placeholder(shape=[None,],
                                          dtype=tf.int64,
                                           name='labels_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO')] + self.labels[:-1]

        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)
        # define the basic cell
        if use_lstm:
            basic_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(emb_dim)
            print("Using LSTM as default cell")
        else:
            basic_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(emb_dim)
            print("Using GRU as default cell")

        # stack cells together : n layered model
        stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


        # for parameter sharing between training model
        #  and testing model
        with tf.variable_scope('decoder') as scope:
            # build the seq2seq model
            #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
            self.decode_outputs,self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,
                                                                                                      self.dec_ip,
                                                                                                      stacked_lstm,
                                                                                                      xvocab_size,
                                                                                                      yvocab_size,
                                                                                                      emb_dim)
            # share parameters
            scope.reuse_variables()
            # testing model, where output of previous timestep is fed as input
            #  to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                feed_previous=True)

        # now, for training,
        #  build loss function
        with tf.name_scope("training"):
            # weighted loss
            #  TODO : adjust weights
            loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs,
                                                                self.labels, loss_weights,
                                                                yvocab_size,
                                                                name="loss")
            tf.summary.scalar("loss", self.loss)
            # train op to minimize the loss
            # TODO: learning rate change?
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, name="train_op")

        self.merge_summary = tf.summary.merge_all()

