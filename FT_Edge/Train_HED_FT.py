import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.HED_Dataset import HED_dataset
import model.HED as HED


Hed_IMG = HED.HED(input_channels=3).cuda()
batch_size = 16
learn_rate = 0.001
train_dst = 'FT_Data\\training\\train_HED_Contains_FT.csv'

train_dataset = HED_dataset(train_dst)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.SGD([{'params': Hed_IMG.parameters()}], lr=learn_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#####################################################################################
# start training
for epoch in range(200):
    epoch_loss_list = [0,0,0,0,0,0,0]
    for i, sample in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        raw_images, images, labels = sample['raw_image'], sample['img'], sample['label']
        img_outputs = Hed_IMG(images)
        
        loss_BS1 = HED.HED_LOSS(img_outputs[0], labels)
        if loss_BS1 == False:
            continue
        loss_BS2 = HED.HED_LOSS(img_outputs[1], labels)
        loss_BS3 = HED.HED_LOSS(img_outputs[2], labels)
        loss_BS4 = HED.HED_LOSS(img_outputs[3], labels)
        loss_BS5 = HED.HED_LOSS(img_outputs[4], labels)
        loss_BM = HED.HED_LOSS(img_outputs[5], labels)

        loss = (0.2*loss_BS1 + 0.2*loss_BS2 + 0.2*loss_BS3 + 0.2*loss_BS4 + 0.2*loss_BS5 + loss_BM)

        epoch_loss_list[0] += loss_BS1.item()
        epoch_loss_list[1] += loss_BS2.item()
        epoch_loss_list[2] += loss_BS3.item()
        epoch_loss_list[3] += loss_BS4.item()
        epoch_loss_list[4] += loss_BS5.item()
        epoch_loss_list[5] += loss_BM.item()
        epoch_loss_list[6] = epoch_loss_list[0]+epoch_loss_list[1]+epoch_loss_list[2]+epoch_loss_list[3]+epoch_loss_list[4]+epoch_loss_list[5]
        
        loss.backward()
        optimizer.step()
    scheduler.step()

    print('--------epoch %d done --------' %epoch)
    print('time: ', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    print('loss_BS1 is %f , loss_BS2 is %f , loss_BS3 is %f , loss_BS4 is %f , loss_BS5 is %f , loss_BM is %f , lossall is %f'
            %(epoch_loss_list[0],epoch_loss_list[1],epoch_loss_list[2],epoch_loss_list[3],epoch_loss_list[4],epoch_loss_list[5],epoch_loss_list[6]))
    print('lr: ', optimizer.param_groups[0]['lr'])
    epoch_loss_list = [0,0,0,0,0,0,0]

    # output temp result
    if epoch % 10 == 0:
        layer = 0
        for i in [img_outputs[0], img_outputs[1], img_outputs[2], img_outputs[3], img_outputs[4], img_outputs[5]]:
            predects = i.cpu().detach().numpy()*255
            predect = predects[0].reshape(predects[0].shape[1],predects[0].shape[2])

            predect = transforms.ToPILImage()(predect)
            predect = predect.convert('RGB')
            predect_fn = 'FT_Edge\\result\\'+str(epoch)+'_'+str(layer)+'pre.png'
            predect.save(predect_fn)

            layer += 1

        labels = labels.cpu().numpy() * 255
        label = labels[0].reshape(predects[0].shape[1],predects[0].shape[2])
        label = transforms.ToPILImage()(label)
        label = label.convert('RGB')
        label_fn = 'FT_Edge\\result\\' + str(epoch) + 'lab.png'
        label.save(label_fn)
        
        img = transforms.ToPILImage()(raw_images[0].cpu())
        img_fn = 'FT_Edge\\result\\' + str(epoch) + 'img.png'
        img.save(img_fn)

    # save model
    if epoch % 10 == 0:
        image_model_name = 'FT_Edge\\result\\HED_model' + str(epoch) + '.pkl'
        torch.save(Hed_IMG, image_model_name)
