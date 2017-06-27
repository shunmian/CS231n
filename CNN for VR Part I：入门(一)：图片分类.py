import numpy as np
import pickle

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10(file):
    results = []
    Xtr = np.zeros([10000,3072])
    Ytr = []
    for i in range(5):
        print("{}/data_batch_{}".format(file,i+1))
        dict = unpickle("{}/data_batch_{}".format(file,i+1))
        Xtr =  np.concatenate((Xtr,dict[b"data"]))
        Ytr = Ytr + dict[b"labels"]
    Ytr = np.array(Ytr)

    dict = unpickle("{}/test_batch".format(file, i + 1))
    Xte = dict[b"data"]
    Yte = dict[b"labels"]

    return (Xtr[59000:],Ytr[49000:],Xte,Yte)

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self,X,y):
        '''
        X is N*D where each row is an exmaple. y is 1-dimension of size N
        '''
        self.Xtr = X
        self.ytr = y

    def predict(self,X):
        '''
        X is N*D where each row is an exmaple we wish to predict label for
        '''
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr-X[i,:]),axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        return Ypred



Xtr,Ytr, Xte,Yte = load_CIFAR10("cifar-10-batches-py")
nn = NearestNeighbor()
nn.train(Xtr,Ytr)
Yte_prdict = nn.predict(Xte)
print("accuracy:{}".format(np.mean(Yte_prdict == Yte)))


