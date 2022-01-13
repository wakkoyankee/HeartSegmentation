import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os

from imageDataGenerator import savePNG, saveBatchPNG
from metrics import confVector, batchConfusionMatrix, recall,saveConfMat,precision,accuracy,f1
from progressBar import printProgressBar
from os.path import isfile, join

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#les classes du mask sont [0.0000, 0.3333, 0.6667, 1.0000] la fonction la rend ces classes sous la forme 0,1,2,3
def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    #print(batch[0,0])
    denom = 0.33333334 # use for ACDC this value
    return (batch / denom).round().long().squeeze()

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return Variable((x == 1).float(), requires_grad=True)


def getOneHotSegmentation(batch):
    backgroundVal = 0
    
    l1 = 0.4
    l2 = 0.7

    b1 = (batch == backgroundVal)
    b2 = (batch < l1) * ~b1
    b3 = (batch < l2) * ~b2 * ~b1
    b4 = (batch > l2)

    oneHotLabels = torch.cat((b1,b2,b3,b4), 
                             dim=1)
                             
    return oneHotLabels.float()

#Dice loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()

    def forward(self, input, target):
        
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()                            
        dice = (2.*intersection + 1e-6)/(input.sum() + target.sum() + 1e-6)    
        return 1 - dice

#Dice + entropie croisée loss
class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):  
        logits = inputs
        ce_target = getTargetSegmentation(targets)
        dice_target = getOneHotSegmentation(targets)

        inputs = F.softmax(inputs, dim=1)

        inputs = inputs.view(-1)
        dice_target = dice_target.view(-1)

        intersection = (inputs * dice_target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + dice_target.sum() + smooth)  
        CE = F.cross_entropy(logits, ce_target,reduction='mean')
        Dice_CE = CE + dice_loss
        return Dice_CE

#fait une inference avec la diceCE loss
def inference(net, img_batch, modelName, epoch, savePNG):
    total = len(img_batch)
    net.eval()

    DiceCE_loss = DiceCELoss().cuda()

    losses = []

    confMat = torch.zeros(4, 4, dtype=torch.int64)
    confVec = torch.zeros(4) # conf vector [TP,TN,FP,FN]

    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)

        DiceCE_loss_value = DiceCE_loss(net_predictions, labels)
        losses.append(DiceCE_loss_value.cpu().data.numpy())


        predOH = predToSegmentation(F.softmax(net_predictions, dim=1))
        labelsOH = getOneHotSegmentation(labels)

        confVec = confVec + confVector(labelsOH,predOH)
        confMat = confMat + batchConfusionMatrix(
            transformBatchOneHot4CtoSC(predOH),
            transformBatchOneHot4CtoSC(labelsOH)
        )

        if savePNG:
            saveBatchPNG(F.softmax(net_predictions,dim=1), "Data/val/Pred/epoch_" + str(epoch) + "/", img_names)


    saveConfMat(confMat.numpy(),"Data/val/")
    print("Accuracy : ", accuracy(confMat).item()*100,"%")
    print("Precision : ", precision(confVec).item()*100,"%")
    print("Recall : ", recall(confVec).item()*100,"%")
    print("F1 : ", f1(confVec).item())
    printProgressBar(total, total, done="[Inference] Metrics Done !")

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()

# Inference pour le calcul de metrics
def inferenceMetrics(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    confMat = torch.zeros(4, 4, dtype=torch.int64)
    confVec = torch.zeros(4) # conf vector [TP,TN,FP,FN]
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)

        predOH = predToSegmentation(F.softmax(net_predictions, dim=1))
        labelsOH = getOneHotSegmentation(labels)

        confVec = confVec + confVector(labelsOH,predOH)
        confMat = confMat + batchConfusionMatrix(
            transformBatchOneHot4CtoSC(predOH),
            transformBatchOneHot4CtoSC(labelsOH)
        )


        # confusion_vector = torch.divide(p, truth)

        # pC = transformBatchOneHot4CtoSC(p)
        # tC = transformBatchOneHot4CtoSC(truth)
        # mat = confusionMatrix(tC[1],pC[1])
        # # test = mat.numpy()
        #
        # test = batchConfusionMatrix(tC,pC).numpy()
        # res = test.sum()
        # test = test/test.sum()*100




        # tp = torch.sum(confusion_vector == 1).item()
        # fp = torch.sum(confusion_vector == float('inf')).item()
        # tn = torch.sum(torch.isnan(confusion_vector)).item()
        # fn = torch.sum(confusion_vector == 0).item()
        #
        # pre = tp / (tp+fp)
        # precision += pre
        # rec = tp / (tp + fn)
        # recall += rec
        # F1tmp = 2 * (precision * recall) / (precision + recall)
        # F1 += F1tmp

        saveBatchPNG(F.softmax(net_predictions, dim=1), "Data/val/Pred/epoch_" + str(epoch) + "/", img_names)


    # precision = precision / total
    # recall = recall / total
    # F1 = F1 / total

    # print("Precision: "+ str(precision))
    # print("Recall: " + str(recall))
    # print("F1: " + str(F1))

    # confMatNP = confMat.numpy()
    # confMatNPPerc = confMatrixPerc(confMat).numpy()
    saveConfMat(confMat.numpy(),"Data/val/")
    print("Accuracy : ", accuracy(confMat).item()*100,"%")
    print("Precision : ", precision(confVec).item()*100,"%")
    print("Recall : ", recall(confVec).item()*100,"%")
    print("F1 : ", f1(confVec).item())
    printProgressBar(total, total, done="[Inference] Metrics Done !")


# to transform a 4 classes one hot encoded matrix in single matrix
def transformOneHot4CtoSC(tensor):
    return tensor[1] + tensor[2]*2 + tensor[3]*3 # classe 0 = 0 in resultant matrix

def transformOneHot4CtoPNG(tensor):
    grayNuance = [0, 0.33, 0.66, 1]
    return tensor[1]*grayNuance[1] + tensor[2]*grayNuance[2] + tensor[3]*grayNuance[3] # classe 0 = 0 in resultant matrix


def transformBatchOneHot4CtoPNG(tensor):
    grayNuance = [0, 0.33, 0.66, 1]
    return tensor[:,1]*grayNuance[1] + tensor[:,2]*grayNuance[2] + tensor[:,3]*grayNuance[3] # classe 0 = 0 in resultant matrix


def transformBatchOneHot4CtoSC(tensor):
    return tensor[:,1] + tensor[:,2]*2 + tensor[:,3]*3 # classe 0 = 0 in resultant matrix
