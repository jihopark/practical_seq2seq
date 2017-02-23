
import tensorflow as tf
import numpy as np
import time

# preprocessed data
from datasets.twitter import data
import data_utils

#model
import seq2seq_wrapper

#training & prediction
from train import train_seq2seq

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("emb_dim", 1000, "Dimensionality of character embedding (default: 1000)")
tf.flags.DEFINE_integer("num_layers", 4, "Number of seq2seq layer (default: 4)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs (default: 100000)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many epochs (default: 500)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")


# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 100, "Set Memory usage percentage (default:100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

tf.logging.set_verbosity(tf.logging.INFO)


# load data from pickle and npy files
print("loading data")
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')

print("done loading. (Q: %s/A: %s) now splitting into train/valid/test" % (idx_q.shape, idx_a.shape))
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = FLAGS.batch_size
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = FLAGS.emb_dim


print("Initialzing model with:")
print("xseq_len=%s, yseq_len=%s" % (xseq_len, yseq_len))
print("xvocab_size=%s, yvocab_size=%s" % (xvocab_size, yvocab_size))
print("emb_dim=%s" % emb_dim)

print("Training with:")
print("training set size=%s" % trainX.shape[0])
print("validiation set size=%s" % validX.shape[0])

timestamp = str(int(time.time()))
out_dir = 'ckpt/twitter/%s' % timestamp
print("Checkpoint saved at %s" % out_dir)

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               emb_dim=emb_dim,
                               num_layers=FLAGS.num_layers,
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

# create session for training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
session_conf = tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=gpu_options)
sess = tf.Session(config=session_conf)
# init all variables


# In[9]:
# sess = model.restore_last_session()
sess = train_seq2seq(model, train_batch_gen,
                    val_batch_gen,
                    sess,
                    FLAGS.num_epochs,
                    FLAGS.checkpoint_every,
                    FLAGS.evaluate_every,
                    ckpt_path=out_dir,
                    id2word_dic=metadata["idx2w"]
                    )
sess.close()
