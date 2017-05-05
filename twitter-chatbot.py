
import tensorflow as tf
import numpy as np
import time

# preprocessed data
from datasets.tweets import data
import data_utils

#model
import seq2seq_wrapper

#training & prediction
from train import train_seq2seq

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("emb_dim", 1000, "Dimensionality of lstm (default:1000)")
tf.flags.DEFINE_integer("num_layers", 1, "Number of seq2seq layer (default: 1)")
tf.flags.DEFINE_boolean("use_lstm", True, "Wheter to use LSTM or GRU")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default:10)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many epochs")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps")
tf.flags.DEFINE_integer("beam_length", 2, "Length of Beam when decoding")
tf.flags.DEFINE_float("learning_rate", 0.5,
                      "Learning Rate of the model(default:0.5")
tf.flags.DEFINE_string("dataset_name", "large", "dataset name")

# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 95, "Set Memory usage percentage (default:95)")
tf.flags.DEFINE_string("logdir", "", "log directory")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

tf.logging.set_verbosity(tf.logging.ERROR)


# load data from pickle and npy files
print("loading data")
train_q, train_a, test_q, test_a, vocab = data.load(FLAGS.dataset_name)
print(train_q.shape)
print(train_a.shape)
print(test_q.shape)
print(test_a.shape)
print(len(vocab["word2id"].keys()))

# parameters
xseq_len = train_q.shape[1]
yseq_len = train_a.shape[1]
batch_size = FLAGS.batch_size
xvocab_size = len(vocab["word2id"].keys()) 
yvocab_size = xvocab_size
emb_dim = FLAGS.emb_dim


print("Initialzing model with:")
print("xseq_len=%s, yseq_len=%s" % (xseq_len, yseq_len))
print("xvocab_size=%s, yvocab_size=%s" % (xvocab_size, yvocab_size))
print("emb_dim=%s" % emb_dim)

print("Training with:")
print("training set size=%s" % train_q.shape[0])
print("test set size=%s" % test_q.shape[0])

logdir = "./logs/"
logdir += str(int(time.time())) if not FLAGS.logdir else FLAGS.logdir
print("Checkpoint saved at %s" % logdir)

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               emb_dim=emb_dim,
                               num_layers=FLAGS.num_layers,
                               learning_rate=FLAGS.learning_rate,
                               use_lstm=FLAGS.use_lstm
                               )

eval_batch_gen = data_utils.rand_batch_gen(test_q, test_a, batch_size)
train_batch_gen = data_utils.rand_batch_gen(train_q, train_a, batch_size)

# create session for training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
session_conf = tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=gpu_options)
sess = tf.Session(config=session_conf)
# init all variables


# In[9]:
# sess = model.restore_last_session()
sess = train_seq2seq(model, train_batch_gen,
                     eval_batch_gen,
                     sess,
                     int(FLAGS.num_epochs*train_q.shape[0]/batch_size),
                     FLAGS.checkpoint_every,
                     FLAGS.evaluate_every,
                     logdir=logdir,
                     vocab=vocab,
                     beam_length=FLAGS.beam_length
                     )
#sess.close()
