import tensorflow as tf
import numpy as np
import sys
import time

import data_utils
from predict import beam_search

# run one batch for training
def train_batch(model, sess, train_batch_gen):
    # get batches
    batchX, batchY = train_batch_gen.__next__()
    # build feed
    feed_dict = get_feed(model, batchX, batchY, keep_prob=1.)
    summary, _, loss_v = sess.run([model.merge_summary,
                             model.train_op,
                             model.loss], feed_dict)
    return summary, loss_v

# get the feed dictionary
def get_feed(model, X, Y, keep_prob):
    # reverse encoder input
    feed_dict = {model.enc_ip[t]: X[model.xseq_len - t - 1] for t in range(model.xseq_len)}

    feed_dict.update({model.labels[t]: Y[t] for t in range(model.yseq_len)})
    feed_dict[model.keep_prob] = keep_prob # dropout prob
    return feed_dict

def eval_step(model, sess, eval_batch_gen):
    # get batches
    batchX, batchY = eval_batch_gen.__next__()
    # build feed
    feed_dict = get_feed(model, batchX, batchY, keep_prob=1.)
    summary, loss_v = sess.run([model.merge_summary,
                                 model.loss], feed_dict)
    # dec_op_v is a list; also need to transpose 0,1 indices
    #  (interchange batch_size and timesteps dimensions
    #dec_op_v = np.array(dec_op_v).transpose([1,0,2])
    return summary, loss_v, batchX, batchY

# evaluate 'num_batches' batches
def eval_batches(model, sess, eval_batch_gen, num_batches, vocab, beam_length):
    losses = []
    outputs = []
    for i in range(num_batches):
        summary, loss_v, batchX, batchY = eval_step(model, sess, eval_batch_gen)
        losses.append(loss_v)

        # sample from random index
        k = np.random.choice(batchX.shape[1], 1)[0]

        beam_output = beam_search(model, sess, batchX[:, k], batchY[:, k],
                                  vocab, B=beam_length, verbose=False)
        outputs.append([batchX[:,k], batchY[:,k], beam_output])

    return summary, np.mean(losses), outputs

def vocab_reverse(vocab, ids):
    return " ".join([vocab["id2word"][id] for id in ids])

def train_seq2seq(model,
                  train_set,
                  eval_set,
                  sess,
                  num_steps,
                  checkpoint_every,
                  evaluate_every,
                  logdir,
                  vocab, beam_length=2):
    if not sess:
        return None

    # we need to save the model periodically
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    eval_writer = tf.summary.FileWriter(logdir + '/eval')


    print('\nTraining started\n')
    print("Total steps=%s" % num_steps)

    for i in range(num_steps):
        try:
            start_time = time.time()
            print(".")
            summary, train_loss = train_batch(model, sess, train_set)
            train_writer.add_summary(summary, i)
            # print loss every 100 steps
            if i and i % 100 == 0:
                print("Step %s: Training loss=%.4f (%s secs)" % (i, train_loss, step_time))
            if i and i % checkpoint_every == 0:
                # save model to disk
                saver.save(sess, logdir + "/" +  model.model_name + '.ckpt', global_step=i)
                print('\nModel saved to disk at iteration #{}'.format(i))

            if i and i % evaluate_every == 0:
                # evaluate to get validation loss
                summary, eval_loss, eval_outputs = eval_batches(model, sess,
                        eval_set, 16, vocab, beam_length)
                eval_writer.add_summary(summary, i)
                # print validation outputs
                print("Validation Output Samples")
                for row in eval_outputs:
                    print(row[0].shape)
                    _input = vocab_reverse(vocab, row[0])
                    _label = vocab_reverse(vocab, row[1])
                    print("q: %s" % _input)
                    print("label: %s" % _label)
                    for n,h in enumerate(row[2]):
                        print("output %s=%s %.10f" % (n, vocab_reverse(vocab, h[0]), h[1]))
                    print("---------")
                # print stats
                print('eval loss: {0:.6f}  perplexity: \
                        {0:.2f}'.format(eval_loss, 2**eval_loss))
                sys.stdout.flush()
            step_time = time.time() - start_time
        except KeyboardInterrupt: # this will most definitely happen, so handle it
            print('Interrupted by user at iteration {}'.format(i))
            break
    train_writer.close()
    eval_writer.close()
    return sess
