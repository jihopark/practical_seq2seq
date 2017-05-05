import tensorflow as tf
import numpy as np

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def beam_search(model, sess, question, label, vocab, B=2, verbose=False):
    y_len = len(label)

    def get_top_b(logits):
        probs = softmax(logits)
        best_b_ids = np.argpartition(probs, -B)[-B:]
        best_b_probs = [probs[id] for id in best_b_ids]
        return best_b_ids, best_b_probs

    def to_string_sequence(ids):
        return " ".join([vocab["id2word"][id] for id in ids])

    def print_beam(beam):
        print("printing beam with size %s" % len(beam))
        for n, h in enumerate(beam):
            print("%s:%s %.5f" %(n, to_string_sequence(h[0]), h[1]))

    def get_feed_decoder(pad_symbol, decoder_input=None):
        y_input = [0]
        if decoder_input:
            assert len(decoder_input) < len(model.dec_ip_beam)
            for i in range(len(decoder_input)):
                y_input.append(decoder_input[i])

        for i in range(len(y_input), len(model.dec_ip_beam)):
            y_input.append(pad_symbol)

        return {model.dec_ip_beam[i]: [y_input[i]] for i in range(len(model.dec_ip_beam))}


    def get_feed_dict(hypothesis=None):
        feed_dict = {model.enc_ip[t]: [question[model.xseq_len - t - 1]] for t in range(model.xseq_len)}
        feed_dict.update(get_feed_decoder(decoder_input=hypothesis, pad_symbol=vocab["word2id"]["<PAD>"]))

        return feed_dict

    def keep_beam_length(_beam):
        if len(_beam) > B:
            _beam = sorted(_beam, key=lambda x: x[1], reverse=True)
            _beam = _beam[:B]
        return _beam

    beam = []
    complete = []
    for i in range(y_len):
        if i == 0:
            dec_op_v = sess.run(model.decode_outputs_test, get_feed_dict())
            dec_op_v = np.array(dec_op_v)

            # add the first partial hypothesis
            ids, probs = get_top_b(dec_op_v[0,0,:])
            for j in range(B):
                beam.append(([ids[j]], probs[j]))
        else:
            if verbose:
                print("\nstarting the search")
                print_beam(beam)
            new_beam = []
            for hypothesis in beam:
                if verbose:
                    print("hypothesis %s" % to_string_sequence(hypothesis[0]))

                dec_op_v = sess.run(model.decode_outputs_test,
                                    get_feed_dict(hypothesis[0]))
                dec_op_v = np.array(dec_op_v)

                ids, probs = get_top_b(dec_op_v[i,0,:])

                for j in range(B):
                    new_sequence = list(hypothesis[0])
                    new_sequence.append(ids[j])
                    new_prob = hypothesis[1] * probs[j]

                    if ids[j] == vocab["word2id"]["<EOS>"] or ids[j] == vocab["word2id"]["<PAD>"]:
                        complete.append((new_sequence, new_prob))
                        if verbose:
                            print("completed hypothesis=%s" %
                                  to_string_sequence(new_sequence))
                    else:
                        new_beam.append((new_sequence,new_prob))
            beam = new_beam
            beam = keep_beam_length(beam)
            if verbose: 
                print("the new beam")
                print_beam(beam)

    complete = keep_beam_length(complete)
    if verbose:
        print("final=")
        print_beam(complete)

    return complete
