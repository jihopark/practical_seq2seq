import os
import numpy as np
import pickle

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
        raise FileNotFoundError("No preprocessing done. See dataset-preprcessing.ipynb")
