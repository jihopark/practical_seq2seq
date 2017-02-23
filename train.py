import tensorflow as tf
import numpy as np
import sys

# run one batch for training
def train_batch(model, sess, train_batch_gen):
    # get batches
    batchX, batchY = train_batch_gen.__next__()
    # build feed
    feed_dict = get_feed(model, batchX, batchY, keep_prob=0.5)
    _, loss_v = sess.run([model.train_op, model.loss], feed_dict)
    return loss_v

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
    loss_v, dec_op_v = sess.run([model.loss, model.decode_outputs_test], feed_dict)
    # dec_op_v is a list; also need to transpose 0,1 indices
    #  (interchange batch_size and timesteps dimensions
    dec_op_v = np.array(dec_op_v).transpose([1,0,2])
    return loss_v, dec_op_v, batchX, batchY

# evaluate 'num_batches' batches
def eval_batches(model, sess, eval_batch_gen, num_batches):
    losses = []
    for i in range(num_batches):
        loss_v, dec_op_v, batchX, batchY = eval_step(model, sess, eval_batch_gen)
        losses.append(loss_v)
        # TODO: show some samples from evaluation
    return np.mean(losses)


def train_seq2seq(model, train_set, valid_set, sess, epochs, checkpoint_every, evaluate_every, ckpt_path):
    if not sess:
        return None

    # we need to save the model periodically
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    sys.stdout.write('\n<log> Training started </log>\n')
    # run M epochs
    for i in range(epochs):
        try:
            train_batch(model, sess, train_set)
            if i and i % checkpoint_every == 0:
                # save model to disk
                saver.save(sess, ckpt_path + model.model_name + '.ckpt', global_step=i)
                print('\nModel saved to disk at iteration #{}'.format(i))

            if i and i % evaluate_every == 0:
                # evaluate to get validation loss
                val_loss = eval_batches(model, sess, valid_set, 16)
                # print stats
                print('val loss: {0:.6f}  perplexity: {0:.2f}'.format(val_loss, 2**val_loss ))
                sys.stdout.flush()
        except KeyboardInterrupt: # this will most definitely happen, so handle it
            print('Interrupted by user at iteration {}'.format(i))
            return sess
