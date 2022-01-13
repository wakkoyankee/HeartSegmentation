import time

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from imageDataGenerator import savePNG
from metrics import confusionMatrix
from progressBar import printProgressBar
from myNetwork import *


import medicalDataLoader
import argparse
import params
from utils import *

import random
import torch



from PIL import Image, ImageOps


import warnings
warnings.filterwarnings("ignore")

def runTraining(args):

    print('Init metrics : ')


    #-------- https://www.kaggle.com/protan/ignite-example ----------

    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    ## Get statistics
    batch_size = args.batch_size #batch size de trainnig
    batch_size_val = args.batch_size_val# de validation

    lr = args.lr
    epoch = args.epochs
    root_dir = 'Data/'  # path to the dataset

    print(' Dataset: {} '.format(root_dir))

    transform = transforms.Compose([#fonction qui transforme image en tensor(utilisé après de trainsetfull)
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=args.augmentDataSet,
                                                      equalize=args.equalize)

    train_loader_full = DataLoader(train_set_full,
                              batch_size=batch_size,
                              worker_init_fn=np.random.seed(0),
                              num_workers=0,
                              shuffle=True)#fonction torch, load les données, mets en batchs, etc...


    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=args.equalize)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=args.equalize)

    # Initialize
    num_classes = args.num_classes
    
    print("~~~~~~~~~~~ Creating the CNN model ~~~~~~~~~~")
    #### Create your own model #####

    net = D_AttU()

    print(" Model Name: {}".format(args.modelName))

    print("Total params: {0:,}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    #### Loss Initialization ####
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = DiceLoss()
    DiceCE_loss = DiceCELoss()

    if torch.cuda.is_available():
        net.cuda()
        CE_loss.cuda()
        Dice_loss.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))

    ### To save statistics ####
    lossTotalTraining = []
    lossTotalValidation = []
    Best_loss_val = 1000
    BestEpoch = 0
    flag = True
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")

    directory = 'Results/Statistics/' + args.modelName

    if os.path.exists(directory)==False:
        os.makedirs(directory)

    tTotal = time.time()
    for i in range(epoch):
        t0 = time.time()
        net.train()
        lossEpoch = []
        num_batches = len(train_loader_full)
        for j, data in enumerate(train_loader_full):#batch par batch
            ### Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()

            images, labels, img_names = data

            ### From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)
        
            ################### Train ###################
            #-- The CNN makes its predictions (forward pass) logits
            net_predictions = net(images)

            #-- Compute the loss --#
            DiceCE_loss_value = DiceCE_loss(net_predictions, labels)

            
            lossTotal = DiceCE_loss_value #

            lossTotal.backward()#donne l'erreur, lance la backprop?
            optimizer.step()

            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, ".format(lossTotal))


        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()#loss final de l'epoch

        lossTotalTraining.append(lossEpoch)#Ajoute la training loss de l'epoch dans la liste

        printProgressBar(num_batches, num_batches,
                             done="[Training] Epoch: {}, LossG: {:.4f}".format(i,lossEpoch))
        if i == epoch-1:
            loss_val = inference(net, val_loader, args.modelName, i, True)#compute la val loss
        else:
            loss_val = inference(net, val_loader, args.modelName, i, args.savePNGeachEP)  # compute la val loss
        lossTotalValidation.append(loss_val)#save les val loss


        np.save(os.path.join(directory, 'Losses.npy'), lossTotalTraining)
        np.save(os.path.join(directory, 'Losses_val.npy'), lossTotalValidation)#ajouter la val loss

        ### Save latest model ####
        
        if(loss_val < Best_loss_val):#Sauvegarde le modele si loss plus petit

            if not os.path.exists('./models/' + args.modelName):
                os.makedirs('./models/' + args.modelName)

            torch.save(net.state_dict(), './models/' + args.modelName + '/' + str(i) + '_Epoch')#save que le meilleur

            Best_loss_val = loss_val
            BestEpoch = i

        print("###                                                       ###")
        print("###  [VAL]  Best Loss : {:.4f} at epoch {}  ###".format(Best_loss_val, BestEpoch))
        print("###                                                       ###")

        if i % (BestEpoch + 10) == 0 and i>0: #si  ca fait 10 epoch qu'on à pas d'amelioration baisser le lr
            for param_group in optimizer.param_groups:
                lr = lr*0.5
                param_group['lr'] = lr
                print(' ----------  New learning Rate: {}'.format(lr))



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelName",default="DAttu",type=str)
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--batch_size_val',default=4,type=int)
    parser.add_argument('--num_classes',default=4,type=int)
    parser.add_argument('--epochs',default=500,type=int)
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--augmentDataSet',default=False,type=bool)
    parser.add_argument('--equalize',default=False,type=bool)
    parser.add_argument('--savePNGeachEP',default=False,type=bool)
    args=parser.parse_args()

    runTraining(args)
