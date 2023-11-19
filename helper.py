import os
import pickle
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

def add_gaussion(X: np.ndarray):
    mu, sigma = 0, 1

    return X + np.random.normal(mu, sigma, X.shape)

def smote_sampling(X: np.ndarray, y: np.ndarray):
    # same
    wanted_distribution = dict(Counter(y))

    sm = SMOTE(sampling_strategy=wanted_distribution)

    return sm.fit_resample(X, y)

def invert_array(X: np.ndarray):
    return np.flip(X, axis=1)

def store_checkpoint(model, name, epoch, prefix='resnet'):
    path2store = os.path.join('./dist/', prefix, f'{name}-{epoch}.pkl')
    with open(path2store, 'wb') as f:
        pickle.dump(model, f)