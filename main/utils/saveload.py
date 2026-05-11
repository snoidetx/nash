import os
import pickle


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
    f.close()


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    f.close()
    return obj
