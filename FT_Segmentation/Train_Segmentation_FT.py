import network
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image
from dataset import segment_dataset
from loss import loss_with_hed

'''
Our DeepLab V3+ Network mainly base on the implementation of https://github.com/VainF/DeepLabV3Plus-Pytorch
We express our sincere thanks
'''

AM_para = 'M'
Out_Filename = 'FT_Segmentation\\result\\Model_SS' + AM_para
batch_size = 8
lr = 0.001
# If AM_para is 'M', then train_M_Seg.csv
train_dst = segment_dataset.deeplabv3plus_dataset('FT_Data\\training\\train_M_Seg.csv') 
train_loader = data.DataLoader(
    train_dst, batch_size=batch_size, shuffle=True)

# Set up model
model_Seg = network.deeplabv3plus_resnet101(input_channels=4, num_classes=3, output_stride=16)
model_HED = torch.load('FT_Edge\\result\\HED_model100.pkl').cuda()

optimizer = torch.optim.SGD(params=[
    {'params': model_Seg.backbone.parameters(), 'lr': 0.1*lr},
    {'params': model_HED.parameters(), 'lr': 0.1*lr},
    {'params': model_Seg.classifier.parameters(), 'lr': lr}
    ], lr=lr, momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#==========   Train Loop   ==========#
cur_itrs = 0
for epoch in range(201):    #epoch number (can be changed)
    # =====  Train  =====
    model_Seg.train()
    model_HED.train()

    interval_loss = 0
    model_Seg = model_Seg.cuda()
    for i, sample in enumerate(train_loader, 0):
        cur_itrs += 1
        optimizer.zero_grad()

        raw_images, images, labels = sample['raw_image'], sample['img'], sample['label']
        HED_out = model_HED(images)

        inputs = torch.cat([images, HED_out[5]], 1)
        out = model_Seg(inputs, HED_out[5])
        if torch.max(labels) == 0:
            continue

        loss = loss_with_hed(out, HED_out[5], labels.long())
        
        loss.backward()
        optimizer.step()

        np_loss = loss.detach().cpu().numpy()
        interval_loss += np_loss
        
    scheduler.step()
    print('--------epoch %d done --------' %epoch)
    print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    print('loss is %f'%(interval_loss))

    if epoch % 1 == 0:
        image_model_name = Out_Filename + '\\seg_image_model' + str(epoch) + '.pkl'
        torch.save(model_Seg, image_model_name)
        hed_model_name = Out_Filename + '\\seg_hed_model' + str(epoch) + '.pkl'
        torch.save(model_HED, hed_model_name)
        
    if epoch >= 0:
        img_fn = Out_Filename + '\\' + str(epoch) + 'img.png'
        lab_fn = Out_Filename + '\\' + str(epoch) + 'lab.png'
        out_img = raw_images.cpu()
        out_img = transforms.ToPILImage()(out_img[0])
        out_img.save(img_fn)

        out_lab = labels.cpu()
        out_lab = transforms.ToPILImage()(out_lab[0])

        out_lab.save(lab_fn)

        flag = 0
        for item in [out]:
            fn = Out_Filename + '\\' + str(epoch) + str(flag) + '.png'
            predects = item[0]
            _, predects = torch.max(item[0], 0)
            predects = predects.int()
            predects = predects.cpu()
            predects = transforms.ToPILImage()(predects)
            predects.save(fn)
            flag += 1
