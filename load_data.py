import os
import numpy as np
import datetime

def load(path):
    starttime = datetime.datetime.now()
    all = np.loadtxt(path, delimiter=",")
    np.random.shuffle(all) #shuffle
    X = all[:, 0:36]
    y = all[:, -1]
    endtime = datetime.datetime.now()
    print("used time:", endtime - starttime)
    #print(X.shape)
    #print(y.shape)
    return X, y

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(PROJECT_ROOT, "data/train.data")
    #path = os.path.join(PROJECT_ROOT, "data/test.data")
    load(path)