import numpy as np
import random
import math
import time
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

#iper-parameters
maxepoch = 100
eta = 0.0001
eta_minus = 0.5
eta_plus = 1.2
RPROP = 'Y'
mode='batch'
stop='GL'
alpha=0.5
strip =2
# mode='online'
# mode='mini'
#stop='PQ'

# activation functions and it s derivative
def relu(x): return np.maximum(0, x)
def drelu(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1
    return y
def sig(x): return 1 / (np.exp(-x) + 1)  # sigmoid function
def dsig(x):  return sig(x) * (1 - sig(x))  # derivative sigmoid function
def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - (tanh(x) ** 2)
def identity(x): return x
def didentity(x): return 1
def sse(x, t): return 0.5 * np.sum(np.sum((x - t) ** 2))  # sum of square error function
def dsse(x, t): return x - t
def softmax(x): return np.exp(x) / sum(np.exp(x))
def crossEntropySM(x, t): return - np.sum(np.sum(t * np.log(softmax(x))))  # cross entropy error function
def dcrossEntropySM(x, t): return softmax(x) - t
sse.__name__ ='SumOfSquare'
crossEntropySM.__name__ ='crossEntropySM'

# net configuration
loss = crossEntropySM
dloss = dcrossEntropySM
afun = sig
dafun = dsig
ofun = identity
dofun = didentity
layers = [100]

def loadData():
    X, y = loadlocal_mnist(
        images_path='./dataset/train-images-idx3-ubyte',
        labels_path='./dataset/train-labels-idx1-ubyte')
    Y = np.zeros((np.unique(y).shape[0], y.shape[0]))
    for t, i in enumerate(Y.T): i[y[t]] = 1
    print("Loading Dataset...Done")
    return X.T, Y
def normalize(x):
    min, max = np.min(x), np.max(x)
    x = (x - min) / ((max - min) + 10 ** -6)
    return x
def split(size, learn):
    train = [i for i in range(size)]
    mode = random.choice((0.5, 0.5))
    valsize = math.floor(mode * size)
    validation = random.sample(train, valsize)
    test = random.sample(validation, math.floor(valsize / 2))
    train = list(set(train) - set(validation))
    validation = list(set(validation) - set(test))
    print(
        f"Slitting dataset..(TrainSet: {(1 - mode) * 100}% ValidationSet: {mode * 50}% TestSet: {mode * 50}%) ...Done")
    if (learn == 'batch'):
        batchSize = len(train)
    elif (learn == 'online'):
        batchSize = 1
    elif (learn == 'mini'):
        batchSize = int((len(train) / 100) * 10)  # mini batch del 10%
    else:
        batchSize = -1, print("{} is not a valid learning mode"), exit(1)
    return train, validation, test, batchSize
def creaRete(x, m, c):
    lay = [x]
    for i in m: lay.append(i)
    lay.append(c)
    features, h, o = lay[0], lay[1], lay[-1]
    x = Net()
    t = h
    x.add(Layer("input", features, t, afun, dafun))
    for r in m[1:]:
        x.add(Layer("hidden", t, r, afun, dafun))
        t = r
    x.add(Layer("output", t, o, ofun, dofun))
    print("Building Neural Network...")
    x.stampa()
    print("Done")
    return x
def showGraph(cm, eTrain, eValidation, accV, precisionValidation, recallValidation, f1Validation, gl, pq):
    fig = plt.figure()
    grid = fig.add_gridspec(2, 3)
    a = fig.add_subplot(grid[0, :])
    b = fig.add_subplot(grid[1, 0])
    c = fig.add_subplot(grid[1, 1])
    d = fig.add_subplot(grid[1, 2])
    a.plot(range(len(eTrain)), eTrain, label="train")
    a.plot(range(len(eValidation)), eValidation, label="validation")
    c.plot(range(len(precisionValidation)), precisionValidation, label="Precision", color='g')
    c.plot(range(len(recallValidation)), recallValidation, label="Recall", color='r')
    b.plot(range(len(f1Validation)), f1Validation, label="F1", color='b')
    b.plot(range(len(accV)), accV, label="Accuracy", color='m')
    d.plot(range(len(gl)), gl, label="GL", color='c')
    d.plot(range(len(pq)), pq, label="PQ", color='y')
    # a.scatter(eValidation.index(min(eValidation)),min(eValidation), s=30,c='green')
    # a.annotate("Min", (eValidation.index(min(eValidation)), min(eValidation)))
    # b.scatter(accV.index(max(accV)),max(accV), s=30,c='blue')
    # b.annotate(f'{round(max(accV),2)}%', (accV.index(max(accV)),max(accV)))
    a.set_ylabel("Error")
    a.set_xlabel("Epoch")
    b.set_ylabel("%")
    b.set_xlabel("Epoch")
    c.set_ylabel("%")
    c.set_xlabel("Epoch")
    d.set_ylabel("%")
    d.set_xlabel("Epoch")
    a.legend()
    b.legend()
    c.legend()
    d.legend()
    plt.show()
    plot_confusion_matrix(cm, colorbar=True, class_names=[f'Digit {i}' for i in range(cm.shape[0])])
    plt.show()

def plot_confusion_matrix(conf_mat, hide_spines=False, hide_ticks=False, figsize=None, cmap=None, colorbar=False,
                          show_absolute=True, show_normed=False, class_names=None):
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            if show_normed:
                ax.text(x=j,
                        y=i,
                        s=cell_text,
                        va='center',
                        ha='center',
                        color="white" if normed_conf_mat[i, j] > 0.5
                        else "black")
            else:
                ax.text(x=j,
                        y=i,
                        s=cell_text,
                        va='center',
                        ha='center',
                        color="white" if conf_mat[i, j] > np.max(conf_mat) / 2
                        else "black")
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax
def earlyStopping(low, high, GL, PQ, alpha, stop, percentage=20):
    if stop == 'GL':
        return low < high and (GL <= alpha or low < high * percentage / 100)
    elif stop == 'PQ':
        return low < high and (PQ <= alpha or low < high * percentage / 100)
    else:
        return low < high
def learn(net, X, Y, mode, stop, alpha, k):
    train, validation, test, batchSize = split(X.shape[1], mode)
    eTrain, eValidation, eTest = [], [], 0
    precision, recall, f1, accuracy = [], [], [], []
    minV = math.inf
    # -----------start early stopping parameters --------
    GL, PQ = 0.0, 0.0  # generalization loss, quotient generalization loss
    gl, pq = [], []  # list of generalization loss, quotient generalization loss
    strip = k  # length for calculate PQ
    PK = []  # progress
    # -----------end early stopping parameters --------
    epoch = 0
    print("Start learning..")
    while earlyStopping(epoch, maxepoch, GL, PQ, alpha, stop):
        train = random.sample(train, len(train))
        validation = random.sample(validation, len(validation))

        for point in range(0, len(train), batchSize):
            net.backprop(X[:, train[point:point + batchSize]], Y[:, train[point:point + batchSize]])

        net.forward(X[:, train])
        eTrain.append(net.getError(Y[:, train]))

        net.forward(X[:, validation])
        eValidation.append(net.getError(Y[:, validation]))
        net.confusionMatrix(Y[:, validation])
        accuracy.append(net.accuracy)
        precision.append(net.precision)
        recall.append(net.recall)
        f1.append(net.f1)

        print("Epoch: {} TL: {} VL: {} Accuracy: {}% Precision: {}% Recall: {}% F1: {}% GL: {}% PQ {}%".
              format(epoch + 1, round(eTrain[epoch], 3), round(eValidation[epoch], 3), round(accuracy[epoch], 3),
                     round(precision[epoch], 3),
                     round(recall[epoch], 3),
                     round(f1[epoch], 3),
                     round(GL, 3), round(PQ, 3)))

        if eValidation[-1] < minV:
            minV = eValidation[-1]
            net.copyNet()

        GL = 100 * (eValidation[-1] / minV - 1)

        if epoch % strip == 0 and epoch > strip:
            PK.append(1000 * (sum(eTrain[epoch - strip:epoch]) / (strip * min(eTrain[epoch - strip:epoch])) - 1))
            PQ = GL / PK[-1]

        gl.append(GL)
        pq.append(PQ)

        epoch += 1

    if stop != 'def' and (GL > alpha or PQ > alpha) : print("Early Stopping...")

    net.forwOnBestNet(X[:, test])
    eTest = net.getError(Y[:, test])
    cm = net.confusionMatrix(Y[:, test])

    print(f"E(train): {round(eTrain[-1],2)}"
          f"\nE(validation): {round(eValidation[-1], 3)}"
          f"\nE(test): {round(eTest, 3)}"
          f"\nPrecision: { round(net.precision, 3)}% "
          f"\nRecall: { round(net.recall, 3)}% "
          f"\nF1: {round(net.f1, 3)} %"
          f"\nAccuracy: {round(net.accuracy, 3)}%"
          f"\nGL: {round(GL, 3)}%"
          f"\nPQ: {round(PQ, 3)}%")


    showGraph(cm, eTrain, eValidation, accuracy, precision, recall, f1, gl, pq)

# layer

class Layer:
    def __init__(self, name, x, m, fun, dfun):
        self.w = 0.1 * np.random.random((m, x))
        self.b = 0.1 * np.random.random((m, 1))
        # -------------------------- for Rprop update -----------------------
        self.derWp, self.derBp = np.full((m, x), 0.1), np.full((m, 1), 0.1)
        self.updateWp, self.updateBp = np.zeros((m, x)), np.zeros((m, 1))
        self.momentWp, self.momentBp = np.full((m, x), 0.0125), np.full((m, 1), 0.0125)
        # -------------------------------------------------------------------
        self.name = name
        self.fun = fun
        self.dfun = dfun
        self.next = None

    def copyLayer(self):
        x = Layer(self.name, 0, 0, self.fun, self.dfun)
        x.w = self.w.copy()
        x.b = self.b.copy()
        if self.next is not None: x.next = self.next.copyLayer()
        return x

    def forwardProp(self, input):
        self.a = np.dot(self.w, input) + self.b
        self.z = self.fun(self.a)
        if self.next is not None:
            return self.next.forwardProp(self.z)
        else:
            return self.z

    def calcolaDelta(self, target):
        if self.next is None:
            self.delta = self.dfun(self.a) * dloss(self.z, target)
        else:
            self.delta = self.dfun(self.a) * self.next.calcolaDelta(target)
        return np.dot(self.w.T, self.delta)

    def update(self, input):
        derW = np.dot(self.delta, input.T)
        derB = np.empty((self.delta.shape[0], 1))
        derB[:, 0] = np.sum(self.delta, axis=1)
        self.w -= eta * derW
        self.b -= eta * derB
        if self.next is not None:
            self.next.update(self.z)

    def rPropUpdate(self, input):
        derW = np.dot(self.delta, input.T)
        pW = derW * self.derWp
        derB = np.empty((self.delta.shape[0], 1))
        derB[:, 0] = np.sum(self.delta, axis=1)
        pB = derB * self.derBp

        mW, mB = np.zeros(np.shape(self.w)), np.zeros(np.shape(self.b))
        idxW, idxB = np.nonzero(pW > 0), np.nonzero(pB > 0)
        max, min = 50, 0

        mW[idxW] = np.minimum(max, eta_plus * self.momentWp[idxW])
        mB[idxB] = np.minimum(max, eta_plus * self.momentBp[idxB])
        idxW, idxB = np.nonzero(pW < 0), np.nonzero(pB < 0)
        mW[idxW] = np.maximum(min, eta_minus * self.momentWp[idxW])
        mB[idxB] = np.maximum(min, eta_minus * self.momentBp[idxB])
        self.w -= np.sign(derW) * mW
        self.b -= np.sign(derB) * mB

        self.derWp, self.derBp = derW, derB
        self.momentWp, self.momentBp = mW, mB

        if self.next is not None:
            self.next.rPropUpdate(self.z)

    def add(self, lay):
        if self.next is not None:
            self.next.add(lay)
        else:
            self.next = lay.next
            self.next = lay

    def setNext(self, next):
        self.next = next

    def setPrev(self, prev):
        self.prev = prev

    def stampa(self):
        s = f' {self.name} ({self.w.shape[1]}) '
        if self.next is not None: s+=self.next.stampa()
        else: s = f'hidden ({self.w.shape[1]}) {self.name} ({self.w.shape[0]}) E: {loss.__name__}'
        return s

# network
class Net:
    def __init__(self):
        self.net = None
        self.out = None
        self.best = None

    def forward(self, input):
        self.out = self.net.forwardProp(input)

    def backprop(self, input, target):
        self.out = self.net.forwardProp(input)
        self.net.calcolaDelta(target)
        self.net.rPropUpdate(input) if RPROP == 'Y' else self.net.update(input)

    def forwOnBestNet(self, input):
        self.out = self.best.forwardProp(input)

    def copyNet(self):
        self.best = self.net.copyLayer()

    def confusionMatrix(self, target):
        actualValues = predictedValues = target.shape[0]
        confusionMatrix = np.zeros((actualValues, predictedValues), dtype=int)
        tp, tn, fp, fn = 0, 1, 2, 3
        precision, recall, f1, accuracy = 0, 1, 2, 3
        tf = np.zeros((actualValues, 4), dtype=int)
        score = np.zeros((actualValues, 4))
        for i, j in zip(self.out.T, target.T): confusionMatrix[np.argmax(j), np.argmax(i)] += 1
        for i in range(actualValues):
            tf[i, tp] = confusionMatrix[i, i]
            tf[i, tn] = confusionMatrix.sum() - confusionMatrix[i, :].sum() - confusionMatrix[:, i].sum() + tf[i, tp]
            tf[i, fp] = confusionMatrix[:, i].sum() - tf[i, tp]
            tf[i, fn] = confusionMatrix[i, :].sum() - tf[i, tp]
            score[i, precision] = tf[i, tp] / (tf[i, tp] + tf[i, fp]) if tf[i, tp] + tf[i, fp] > 0 else 0
            score[i, recall] = tf[i, tp] / (tf[i, tp] + tf[i, fn]) if tf[i, tp] + tf[i, fn] > 0 else 0
            score[i, f1] = 2 * ((score[i, precision] * score[i, recall]) / (score[i, precision] + score[i, recall])) if \
            score[i, precision] + score[i, recall] > 0 else 0
            score[i, accuracy] = tf[i, tp] / target[i, :].sum()
        self.precision = score[:, precision].sum() / actualValues * 100
        self.recall = score[:, recall].sum() / actualValues * 100
        self.f1 = score[:, f1].sum() / actualValues * 100
        self.accuracy = score[:, accuracy].sum() / actualValues * 100
        return confusionMatrix

    def computeAccuracy(self, target):
        accuracy = 0
        for i, j in zip(self.out.T, target.T):
            if np.argmax(i) == np.argmax(j):
                accuracy += 1
        return accuracy / target.shape[1] * 100

    def getError(self, target):
        return loss(self.out, target)

    def add(self, lay):
        if (self.net is None):
            lay.next = self.net
            self.net = lay
        else:
            self.net.add(lay)

    def stampa(self):
        print(self.net.stampa())



start_time = time.time()

X, Y = loadData()
X = normalize(X)
Y = normalize(Y)

x = creaRete(X.shape[0], layers, Y.shape[0])
learn(x, X, Y, mode, stop,alpha,strip)

print("\n\n--- %lf seconds ---" % (time.time() - start_time))
