import os
import random
from collections import Counter
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

def preprocess(name, max_len):
    def read_lines(filenames):
        data_files = []
        for filename in filenames:
            print("Reading Tweets_processed_%s.txt" % filename)
            f = pd.read_csv('./Tweets_processed_%s.txt' % filename,
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
        for i, row in tqdm(data_files.iterrows()):
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

    # 1. load the whole dataset
    q, a = read_lines(["Train", "Valid", "Test"])

    # 2. shuffle the dataset
    random_idx = list(range(len(q)))
    random.shuffle(random_idx)
    train_idx = random_idx[:int(len(random_idx)*0.9)]
    test_idx = random_idx[int(len(random_idx)*0.9)+1:]
    print(len(train_idx))
    print(len(test_idx))

    for i in test_idx[:10]:
        print("Q: %s A: %s\n" % (q[i], a[i]))

    # 3. split the dataset into train, test
    test_q = [q[i] for i in test_idx]
    test_a = [a[i] for i in test_idx]
    train_q = [q[i] for i in tqdm(train_idx)]
    train_a = [a[i] for i in tqdm(train_idx)]
    print("splitted the dataset into two, %s/%s" % (len(train_q), len(test_q)))

    # 4. split into tokens
    tokens_test_q = [line.split(" ") for line in tqdm(test_q)]
    tokens_test_a = [line.split(" ") for line in tqdm(test_a)]

    tokens_train_q = [line.split(" ") for line in tqdm(train_q)]
    tokens_train_a = [line.split(" ") for line in tqdm(train_a)]
    print("tokenized the dataset")

    # 5. filter out txt by length
    train_tuple = []
    for i in range(len(tokens_train_q)):
        train_tuple.append((tokens_train_q[i], tokens_train_a[i]))
    test_tuple = []
    for i in range(len(tokens_test_q)):
        test_tuple.append((tokens_test_q[i], tokens_test_a[i]))

    _test = list(filter(lambda x: len(x[0]) >= 3 and len(x[0]) <= max_len - 1 and len(x[1]) >= 3 and len(x[1]) <= max_len - 1, test_tuple))
    _train = list(filter(lambda x: len(x[0]) >= 3 and len(x[0]) <= max_len - 1 and len(x[1]) >= 3 and len(x[1]) <= max_len - 1, train_tuple))

    print("train filtered from %s to %s" % (len(train_tuple), len(_train)))
    print("test filtered from %s to %s" % (len(test_tuple), len(_test)))

    print(_train[10])

    # 5. add <EOS> and <PAD> special tokens
    def add_special_tokens(pairs):
        for q,a in tqdm(pairs):
            a.append("<EOS>")
            for _ in range(max_len - len(a)):
                a.append("<PAD>")
            for _ in range(max_len - len(q)):
                q.append("<PAD>")
    add_special_tokens(_test)
    add_special_tokens(_train)

    print("added special tokens")
    print(_train[10])


    # 6. save the file
    _train_q = [" ".join(x[0]) for x in tqdm(_train)]
    _train_a = [" ".join(x[1]) for x in tqdm(_train)]
    assert len(_train_q) ==  len(_train_a)

    _test_q = [" ".join(x[0]) for x in tqdm(_test)]
    _test_a = [" ".join(x[1]) for x in tqdm(_test)]
    assert len(_test_q) == len(_test_a)

    def save_file(name, q, a):
        with open("%s.question" % name, "w") as f1:
            with open("%s.answer" % name, "w") as f2:
                for i in range(len(q)):
                    f1.write(q[i].replace("\n", " ") + "\n")
                    f2.write(a[i].replace("\n", " ") + "\n")
    save_file("train_" + name, _train_q, _train_a)
    save_file("test_" + name, _test_q, _test_a)

def load_preprocessed_file(name):
    with open("%s.question" % name, "r") as f1:
        with open("%s.answer" % name, "r") as f2:
            q = [line.rstrip() for line in f1]
            a = [line.rstrip() for line in f2]
            assert len(q) == len(a)
    return q,a

def create_vocab_and_transform(name, vocab_size):
    # 1. load from file
    train_q, train_a = load_preprocessed_file("train_" + name)
    test_q, test_a = load_preprocessed_file("test_" + name)

    # 2. tokenize
    tokens_test_q = [line.split(" ") for line in tqdm(test_q)]
    tokens_test_a = [line.split(" ") for line in tqdm(test_a)]

    tokens_train_q = [line.split(" ") for line in tqdm(train_q)]
    tokens_train_a = [line.split(" ") for line in tqdm(train_a)]
    print("tokenized the dataset")

    # 3. merge all the words
    total_list = []
    for q,a in tqdm(zip(tokens_test_q, tokens_test_a)):
        total_list.extend(q)
        total_list.extend(a)
    for q,a in tqdm(zip(tokens_train_q, tokens_train_a)):
        total_list.extend(q)
        total_list.extend(a)
    print("total number of tokens=%s" % len(total_list))

    # 4. create vocab
    vocab = Counter(total_list)
    print(vocab.most_common(10))

    vocabulary = {"GO":0, "UNK": 1}
    reverse_vocabulary = {0:"GO", 1:"UNK"}
    count = 2
    for word, _ in vocab.most_common(vocab_size):
        vocabulary.update({word: count})
        reverse_vocabulary.update({count:word})
        count += 1
    print("vocab created with length=%s" % len(vocabulary.keys()))

    # 5. transform
    def transform(q_list, a_list):
        q_vocab = []
        a_vocab = []

        error = 0
        for q,a in tqdm(zip(q_list, a_list)):
            if len(q) == len(a):
                q_vocab.append([vocabulary[token] if token in vocabulary else 1 for token in q])
                a_vocab.append([vocabulary[token] if token in vocabulary else 1 for token in a])
            else:
                error += 1
        print("transform complete with error rate %.3f" % float(error/len(q_list)))
        q_vocab = np.array(q_vocab)
        a_vocab = np.array(a_vocab)
        assert len(q_vocab.shape) == 2 and len(a_vocab.shape) == 2
        assert q_vocab.shape[0] == a_vocab.shape[0]
        assert q_vocab.shape[1] == a_vocab.shape[1]
        print("vocab matrix shape q:%s a:%s" % (str(q_vocab.shape), str(a_vocab.shape)))

        return q_vocab, a_vocab

    test_q_vocab, test_a_vocab = transform(tokens_test_q, tokens_test_a)
    train_q_vocab, train_a_vocab = transform(tokens_train_q, tokens_train_a)

    # 6. save file
    np.save('train_q_%s.npy' % name, train_q_vocab)
    np.save('train_a_%s.npy' % name, train_a_vocab)

    np.save('test_q_%s.npy' % name, test_q_vocab)
    np.save('test_a_%s.npy' % name, test_a_vocab)

    vocab_ = {"word2id": vocabulary, "id2word":reverse_vocabulary}

    with open('%s.vocab' % name, 'wb') as f:
        pickle.dump(vocab_, f)


def load(name):
    file_path = os.path.dirname(os.path.abspath(__file__))
    print("file path=" + file_path)
    with open(file_path + "/%s.vocab" % name, "rb") as f:
        vocab = pickle.load(f)
    try:
        return (np.load(file_path + "/train_q_%s.npy" % name),
                np.load(file_path + "/train_a_%s.npy" % name),
                np.load(file_path + "/test_q_%s.npy" % name),
                np.load(file_path + "/test_a_%s.npy" % name),
                vocab)
    except:
        raise FileNotFoundError("No preprocessing done. Run data.py")

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--vocabulary", action="store_true", default=False)
    parser.add_argument("--max_len", action="store", default=0, type=int)
    parser.add_argument("--vocab_size", default=0, type=int)
    parser.add_argument("--name", type=str, required=True)
    args = vars(parser.parse_args())

    name = args["name"]
    if args["preprocess"]:
        max_len = args["max_len"]
        if max_len <= 0:
            raise ValueError("Please put --max_len and value")

        print("starting preprocessing. name=%s/max_len=%s" % (name, max_len))
        preprocess(name, max_len)
    elif args["vocabulary"]:
        vocab_size = args["vocab_size"]
        if vocab_size <= 0:
            raise ValueError("Please put --vocab_size and value")
        print("starting creating vocab. name=%s/vocab_size=%s" % (name, vocab_size))
        create_vocab_and_transform(name, vocab_size)
    else:
        raise ValueError("Please use --preprocess or --vocabulary flags")
