import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from Voc_loader import VOCLoader
from Unet_model import Unet
from metrics import custom_conf_matrix

def train():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = data.DataLoader(
        VOCLoader("./",do_transform=True),
        shuffle=True,
        batch_size=1,
        #num_workers=8,
    ) 

    val_loader = data.DataLoader(
        VOCLoader("./",portion="val",do_transform=True),
        batch_size=1,
        #num_workers=8,
    ) 

    model = Unet()
    print(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=5*1e-4, lr = 0.00001, momentum=0.9)

    best_iou = -100
    epoch = 0
    total_epochs = 20
    train_step=5
    val_step =1
    #cur_avg_loss = 0
    train_losses = {}
    #train_avg_losses= {} 
    val_losses = {}
    model.cuda()
    while(epoch<total_epochs):
        epoch_loss=0
        for (imgs,labels) in train_loader:
            model.train()            
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out,labels)
            #out_pred = out.data.max(1)[1].cpu().numpy()
            #print(out_pred.shape,out.shape,labels.shape)
            
            loss.backward()
            optimizer.step()
            
            #cur_avg_loss = max([0,epoch])*cur_avg_loss + loss.item()
            #cur_avg_loss /= (iter+1)
            epoch_loss+=loss.item()
        train_losses[epoch] = epoch_loss#loss.item()
        #train_avg_losses[iter] = cur_avg_loss
        if epoch % train_step==0:
            print("iter:",epoch," loss:",epoch_loss)
        if epoch % val_step==0: #or (iter+1)==total_iters:
            print("val_step")
            model.eval()
            conf_mat = custom_conf_matrix([i for i in range(0,21)],21)
            with torch.no_grad():
                val_loss=0
                for vi, (vimg,vlbl) in enumerate(tqdm(val_loader)):
                    vimg, vlbl = vimg.to(device), vlbl.to(device)
                    vout = model(imgs)
                    
                    vloss = criterion(vout,vlbl)
                    pred = vout.data.max(1)[1].cpu().numpy()
                    gt = vlbl.data.cpu().numpy()
                    conf_mat.update_step(gt.flatten(), pred.flatten())
                    val_loss += vloss.item()
                print("epoch:",epoch," val loss:",val_loss,"mean iou ",conf_mat.compute_mean_iou())
                val_losses[epoch] = val_loss
        epoch+=1            
    print(train_losses,val_losses)
if __name__ == "__main__":
   
    #run_id = random.randint(1, 100000)
    train()