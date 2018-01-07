import numpy as np


def get_accuracy_logistic(Yhat, Y):
    predictions = Yhat > 0.5
    correct = (predictions == Y)
    return np.mean(correct)


def get_precision_recall_logistic(Yhat, Y):
    p = Yhat > 0.5
    assert(p.shape == Yhat.shape == Y.shape)
    assert(p.shape[1] == 1)
    tp = np.sum(np.multiply(p == 1, Y == 1))
    fp = np.sum(np.multiply(p == 1, Y == 0))
    fn = np.sum(np.multiply(p == 0, Y == 1))

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    return precision, recall


def get_F1_logistic(Yhat, Y):
    precision, recall = get_precision_recall_logistic(Yhat, Y)
    F1 = (2 * recall * precision) / (recall + precision)
    return F1, precision, recall