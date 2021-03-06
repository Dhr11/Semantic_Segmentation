"""
Creator:
Dhruuv Agarwal
Github: Dhr11
"""

import os
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
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=5*1e-4, lr = 0.0001, momentum=0.9)

    best_iou = -100
    epoch = 0
    total_epochs =100000 
    train_step=5
    val_step =10
    #cur_avg_loss = 0
    train_losses = {}
    #train_avg_losses= {} 
    val_losses = {}
    val_iou = {}
    train_iou = {}
    iou_interval = val_step*2
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
        train_losses[epoch] = epoch_loss/len(train_loader)#loss.item()
        #train_avg_losses[iter] = cur_avg_loss
        if epoch % train_step==0:
            print("epoch:",epoch," loss:",epoch_loss/len(train_loader))
        if epoch % val_step==0: #or (iter+1)==total_iters:
            calc_iou = epoch % iou_interval==0
            print("val_step")
            model.eval()
            conf_mat = custom_conf_matrix([i for i in range(0,21)],21)
            with torch.no_grad():
                val_loss=0
                for vi, (vimg,vlbl) in enumerate(tqdm(val_loader)):
                    vimg, vlbl = vimg.to(device), vlbl.to(device)
                    vout = model(imgs)
                    
                    vloss = criterion(vout,vlbl)
                    if calc_iou:
                        pred = vout.data.max(1)[1].cpu().numpy()
                        gt = vlbl.data.cpu().numpy()
                        conf_mat.update_step(gt.flatten(), pred.flatten())
                    val_loss += vloss.item()
                val_losses[epoch] = val_loss/len(val_loader)
                
                if calc_iou:
                    score = conf_mat.compute_mean_iou()
                    print("epoch:",epoch," val loss:",val_loss/len(val_loader),"mean iou ",score)
                    if score>best_iou:
                        best_iou = score
                        state = {
                            "epoch": epoch + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            "./",
                            "{}_epoch{}_best_model.pkl".format("Unet_pascalVOC", epoch),
                        )
                        torch.save(state, save_path)
                else:
                    print("epoch:",epoch," val loss:",val_loss/len(val_loader))            
                conf_mat.reset()
        epoch+=1            
    print(train_losses,val_losses,val_iou)
if __name__ == "__main__":
   
    #run_id = random.randint(1, 100000)
    train()