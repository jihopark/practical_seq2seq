
import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.twitter import data
import data_utils


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 1024, "Dimensionality of character embedding (default: 1024)")
tf.flags.DEFINE_integer("num_layers", 3, "Number of seq2seq layer (default: 3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs (default: 10000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many epochs (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = FLAGS.batch_size
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = FLAGS.num_epochs

import seq2seq_wrapper

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=FLAGS.num_layers,
                               epochs=FLAGS.num_epochs,
                               evaluate_every=FLAGS.evaluate_every,
                               checkpoint_every=FLAGS.checkpoint_every
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
