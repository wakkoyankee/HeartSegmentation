import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt



def confusionMatrix(labelsTensor, predsTensor):
    labelsTensor = labelsTensor.view(-1).type(torch.int64)
    predsTensor = predsTensor.view(-1).type(torch.int64)
    stacked = torch.stack(
        (
            labelsTensor
            , predsTensor
        )
        , dim=1
    )
    confusionMatrix = torch.zeros(4, 4, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        confusionMatrix[tl, pl] = confusionMatrix[tl, pl] + 1
    return confusionMatrix


def batchConfusionMatrix(labelsTensorB, predsTensorB):
    confusionMatrix = torch.zeros(4, 4, dtype=torch.int64)
    for i in range(len(labelsTensorB)):
        labelsTensor = labelsTensorB[i].view(-1).type(torch.int64)
        predsTensor = predsTensorB[i].view(-1).type(torch.int64)
        stacked = torch.stack(
            (
                labelsTensor
                , predsTensor
            )
            , dim=1
        )
        for p in stacked:
            tl, pl = p.tolist()
            confusionMatrix[tl, pl] = confusionMatrix[tl, pl] + 1

    return confusionMatrix

# conf matrix + conf vector [TP,TN,FP,FN]
def batchCM_CV(labelsTensorB, predsTensorB):
    confusionMatrix = torch.zeros(4, 4, dtype=torch.int64)
    confVec = torch.zeros(4, dtype=torch.int64)  # conf vector [TP,TN,FP,FN]
    for i in range(len(labelsTensorB)):
        labelsTensor = labelsTensorB[i].view(-1).type(torch.int64)
        predsTensor = predsTensorB[i].view(-1).type(torch.int64)
        stacked = torch.stack(
            (
                labelsTensor
                , predsTensor
            )
            , dim=1
        )
        for p in stacked:
            tl, pl = p.tolist()
            confusionMatrix[tl, pl] = confusionMatrix[tl, pl] + 1

    return confusionMatrix,


# to convert a matrix in a occurence percentage matrix
def confMatrixPerc(mat):
    return mat / mat.sum() * 100

def saveConfMat(confMat, path):
    cmap = plt.cm.Blues
    classes = ["Autre","Ventricule droit","Myocarde","Ventricule Gauche"]
    title = 'Matrice de confusion normalisée'
    confMat = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]

    plt.imshow(confMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path + "Matrice_confusion.png")

def confVector(labelsTensor, predsTensor):
    confusion_vector = torch.divide(predsTensor, labelsTensor)
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    return torch.tensor([tp,tn,fp,fn]) # conf vector [TP,TN,FP,FN]


def f1(confVec):
    return (precision(confVec) * recall(confVec)) / (precision(confVec) + recall(confVec))

def accuracy(confMat):
    return confMat.trace() / confMat.sum()

def precision(confVec):
    return confVec[0] / (confVec[0] + confVec[2])

def recall(confVec):
    return confVec[0] / (confVec[0] + confVec[3])