import tensorflow as tf
import numpy as np

def restore_last_session():
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction
    # TODO: rescoring
    def predict(model, sess, X):
        feed_dict = {model.enc_ip[t]: X[t] for t in range(model.xseq_len)}
        feed_dict[model.keep_prob] = 1.
        dec_op_v = sess.run(model.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)
