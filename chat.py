
import tensorflow as tf
import numpy as np
import time
import pickle

# preprocessed data
from datasets.tweets import data
import data_utils

# model
import seq2seq_wrapper

#training & prediction
from train import train_seq2seq
from predict import beam_search


# Parameters
# ==================================================

# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 95, "Set Memory usage percentage (default:95)")
tf.flags.DEFINE_string("logdir", "pretrained_models/two", "log directory")
tf.flags.DEFINE_string("dataset_name", "large", "dataset_name")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

tf.logging.set_verbosity(tf.logging.ERROR)

logdir = "./" + FLAGS.logdir

print("loading model from " + logdir)

with open(logdir + "/model.meta", "rb") as f:
    metadata = pickle.load(f)

# parameters
xseq_len = metadata["xseq_len"]
yseq_len = metadata["yseq_len"]
xvocab_size = metadata["xvocab_size"]
yvocab_size = metadata["yvocab_size"]
emb_dim = metadata["emb_dim"]
use_lstm = metadata["use_lstm"]
num_layers = metadata["num_layers"]

print("Initialzing model with:")
print("xseq_len=%s, yseq_len=%s" % (xseq_len, yseq_len))
print("xvocab_size=%s, yvocab_size=%s" % (xvocab_size, yvocab_size))
print("emb_dim=%s" % emb_dim)
print("num_layers=%s" % num_layers)

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                emb_dim=emb_dim,
                                num_layers=num_layers,
                                use_lstm=use_lstm
                                )

vocab = data.load_vocab(FLAGS.dataset_name)

print("loaded vocabulary")
assert len(vocab["word2id"].keys()) == xvocab_size

print(len(vocab["word2id"].keys()))

saver_path = logdir
checkpoint_file = tf.train.get_checkpoint_state(saver_path)
ckpt_path = checkpoint_file.model_checkpoint_path

print("restoring tensorflow ckpt from %s" %
      checkpoint_file.model_checkpoint_path)

# create session for training
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage / 100)
session_conf = tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=gpu_options)
sess = tf.Session(config=session_conf)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file.model_checkpoint_path)


def encode_input(question):
    _question = []
    tokens = question.split(" ")
    if len(tokens) > xseq_len:
        tokens = tokens[:xseq_len]
    for token in tokens:
        if token in vocab["word2id"].keys():
            _question.append(vocab["word2id"][token])
        else:
            _question.append(vocab["word2id"]["UNK"])

    for _ in range(yseq_len - len(_question)):
        _question.append(vocab["word2id"]["<PAD>"])

    return _question


def answer(question, beam_length=2, use_random=True):
    encoded = encode_input(question)
    output1 = beam_search(model, sess, encoded, np.ones_like(encoded),
                       vocab, use_random, B=beam_length, decode_output=True, verbose=False)
    return output1


print(answer("who are you?"))


def close_session():
    sess.close()
