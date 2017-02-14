EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz !.,' # space and punctuation marks are included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 3
        }

UNK = 'unk'
VOCAB_SIZE = 6000

import random
import sys

import nltk
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

import pickle


def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filenames):
    data_files = []
    for filename in filenames:
        print("Reading Tweets_processed_%s.txt" % filename)
        f = pd.read_csv('./raw_data/Tweets_processed_%s.txt' % filename,
                         sep="\t",
                         skiprows=[0],
                         error_bad_lines=False,
                         names=["index", "tweet_id", "text", "user_id"],
                         dtype={"text": str, "user_id": str, "tweet_id": str, "index": str})
        data_files.append(f)


    data_files = pd.concat(data_files)
    print("Appending all the files")

    count = 0
    q_lines = []
    a_lines = []
    previous = None
    for i, row in data_files.iterrows():
        try:
            index = int(row['index'])
        except ValueError:
            print("ValueError %s" % row['index'])
            index = ""
        if index:
            if previous != None:
                if previous[0] == index - 1:
                    q_lines.append(str(previous[1]))
                    a_lines.append(str(row['text']))
                    count += 1
            previous = (index, row['text'])

    print("%s Q&A pairs added" % count)
    return q_lines, a_lines
'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(q, a):
    filtered_q, filtered_a = [], []
    raw_data_len = len(q)

    for i in range(len(q)):
        qlen = len(q[i])
        alen = len(a[i])
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(q[i])
                filtered_a.append(a[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a





'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():

    print('\n>> Read lines from file')
    q_lines, a_lines = read_lines(["Test", "Train", "Valid"])

    # filter out unnecessary characters
    print('\n>> Filter lines')
    q_lines = [ filter_line(line, EN_WHITELIST) for line in q_lines ]
    a_lines = [ filter_line(line, EN_WHITELIST) for line in a_lines ]

    print(q_lines[200:202])
    print(a_lines[200:202])

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    q_lines, a_lines = filter_data(q_lines, a_lines)

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = map(lambda x: tknzr.tokenize(x),  q_lines)
    atokenized = map(lambda x: tknzr.tokenize(x),  a_lines)


    print('\n:: Sample from segmented list of words')
    print('\nq : {0} ; a : {1}'.format(q_lines[100], a_lines[101]))
    print('\nq : {0} ; a : {1}'.format(q_lines[200], a_lines[201]))

    print('\nq : {0} ; a : {1}'.format(qtokenized[100], atokenized[101]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[200], atokenized[201]))

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


if __name__ == '__main__':
    process_data()
