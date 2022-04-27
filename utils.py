from sklearn.metrics import roc_auc_score
import numpy as np

def arr2hot(arr, N):
    res = [0] * N
    for e in arr:
        res[e - 1] = 1
    return res

def evaluate(pred, y):
    bs = pred.shape[0]
    auc = roc_auc_score(y, pred)
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(bs):
        if pred[i] == y[i]:
            if pred[i] == 1:
                TP += 1
            else:
                TN += 1
        elif pred[i] == 1:
            FP += 1
        else:
            FN += 1
    print('total predict num: {}, correct predict: {}, wrong predict: {}'.format(TP + FP + TN + FN, TP + TN, FP + FN))
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP, TN, FP, FN))
    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('acc: {}, auc: {}, precision: {}, recall: {}, f1: {}, rmse: {}'.format(acc, auc, precision, recall, f1, rmse))
    return acc, auc, precision, recall, f1, rmse