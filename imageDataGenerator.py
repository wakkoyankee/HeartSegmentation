import os
import torch
from torchvision.utils import save_image


def savePNG(tensor, dir, fileName):
    if(not(os.path.exists("Data/val/Pred/"))):
        os.mkdir("Data/val/Pred/")
    if(not(os.path.exists(dir))):
        os.mkdir(dir)
    return save_image(tensor, dir + fileName)

# tensors needs to be soft maxed before
def saveBatchPNG(tensors, dir, fileNames):
    for i in range(len(tensors)):
        if(len(tensors[i])>1):
            savePNG(transformMCtoSC(tensors[i]), dir, os.path.basename(fileNames[i]))
        else:
            savePNG(tensors[i], dir, fileNames[i])


# to concatenate prediction classes into one class
# pue la merde à faire en fonction mé g pa trouvé
def transformMCtoSC(tensor):
    # w = width l = length c = class
    # image = generated img 0= bckgrnd 1=..
    test = torch.zeros((len(tensor[0]), len(tensor[0][0])))
    image = torch.zeros((len(tensor[0]), len(tensor[0][0])))
    grayNuance = [0,0.33,0.66,1]
    for w in range(len(tensor[0])):
        for l in range(len(tensor[0][0])):
            for c in range(len(tensor)):
                if(test[w][l] < tensor[c][w][l]):
                    test[w][l] = tensor[c][w][l]
                    image[w][l] = grayNuance[c]
            # if(image[w][l] != 0):
            #     print('w:',w,' l:',l,' 0:',tensor[0][w][l],' 1:',tensor[1][w][l],' 2:',tensor[2][w][l],' 3:',tensor[3][w][l], ' val : ', image[w][l])
    return image