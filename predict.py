import tensorflow as tf
import numpy as np
import random

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
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum()
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def beam_search(model, sess, question, label, vocab, use_random=True, B=2,
                decode_output=False, verbose=False):
    y_len = len(label)

    def get_next_tokens(logits, is_first=False):
        probs = softmax(logits)
        ids = []
        if use_random and not is_first:
            try:
                while len(ids) != B: # repeat toss until we get B result
                    ids = []
                    random_toss = np.random.multinomial(B, probs)
                    for id, value in enumerate(random_toss):
                        if value > 0 and id != vocab["word2id"]["UNK"]:
                            ids.append(id)
            except ValueError:
                #TODO: find why we get the softmax miscalculation.
                # it doesn't add up to one
                # for now it just chooses the most probable
                ids = np.argpartition(probs, -B)[-B:]
        else:
            ids = np.argpartition(probs, -B)[-B:]
        best_b_probs = [probs[id] for id in ids]
        return ids, best_b_probs

    def to_string_sequence(ids):
        return " ".join([vocab["id2word"][id] for id in ids])

    def print_beam(beam):
        for n, h in enumerate(beam):
            print("%s:%s %.5f" %(n, to_string_sequence(h[0]), h[1]))

    def decode_beam(beam):
        decoded = []
        for seq, prob in beam:
            decoded.append((to_string_sequence(seq), prob))
        return decoded

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

    def keep_beam_length(_beam, use_random=False):
        if use_random:
            return random.sample(_beam, B)
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
            ids, probs = get_next_tokens(dec_op_v[0,0,:])
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

                ids, probs = get_next_tokens(dec_op_v[i,0,:], is_first=True)

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

    complete = keep_beam_length(complete, use_random)
    if verbose:
        print("final=")
        print_beam(complete)

    return decode_beam(complete) if decode_output else complete
