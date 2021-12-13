import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class HED(nn.Module):
    def __init__(self, input_channels):
        super(HED,self).__init__()
        self.input_channels = input_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv_4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv_5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)


        self.sideout_conv1 = nn.Conv2d(64, 1, 1)
        self.sideout_conv2 = nn.Conv2d(128, 1, 1)
        self.sideout_conv3 = nn.Conv2d(256, 1, 1)
        self.sideout_conv4 = nn.Conv2d(512, 1, 1)
        self.sideout_conv5 = nn.Conv2d(512, 1, 1)
        self.sideout_fuse = nn.Conv2d(5, 1, 1)

        self.Combine = torch.nn.Sequential(
			nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        # Initializing HED with VGG pretrained model
        if self.input_channels == 3:
            HED_load_premodel(self, 'FT_Edge\\pretrained\\vgg16-397923af.pth')

    def forward(self, x):

        img_h = x.size(2)
        img_w = x.size(3)

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        x5 = self.conv_5(x4)

        sideout_1 = self.sideout_conv1(x1)
        sideout_2_upsample = F.interpolate(self.sideout_conv2(x2), size=(img_h, img_w), mode='bilinear')
        sideout_3_upsample = F.interpolate(self.sideout_conv3(x3), size=(img_h, img_w), mode='bilinear')
        sideout_4_upsample = F.interpolate(self.sideout_conv4(x4), size=(img_h, img_w), mode='bilinear')
        sideout_5_upsample = F.interpolate(self.sideout_conv5(x5), size=(img_h, img_w), mode='bilinear')

        sideout_concat = self.sideout_fuse(torch.cat((sideout_1,sideout_2_upsample,sideout_3_upsample,sideout_4_upsample,sideout_5_upsample), 1))

        sideout1 = torch.sigmoid(sideout_1)
        sideout2 = torch.sigmoid(sideout_2_upsample)
        sideout3 = torch.sigmoid(sideout_3_upsample)
        sideout4 = torch.sigmoid(sideout_4_upsample)
        sideout5 = torch.sigmoid(sideout_5_upsample)
        sideoutcat = torch.sigmoid(sideout_concat)

        return sideout1, sideout2, sideout3, sideout4, sideout5, sideoutcat

def HED_LOSS(input, target):
    n, c, h, w = input.size()

    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    if pos_num==0:
        return False
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num
    weight = weight/np.max(weight)
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    # print(log_p.shape, target_t.shape)
    loss = F.binary_cross_entropy(log_p, target_t.float(), weight, size_average=True)
    return loss


def HED_load_premodel(model, premodel_filename):
    new_params = torch.load(premodel_filename)
    model_dict = model.state_dict()
    premodel_dict = new_params
    premodel_list = []
    for key, value in premodel_dict.items():
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)
    param_layer = 0
    for key in model_dict:
        if 'deconv' in key:
            break
        if 'conv' in key:
            pre_k = premodel_list[param_layer]['key']
            pre_v = premodel_list[param_layer]['value']
            assert model_dict[key].shape == pre_v.shape
            model_dict[key] = pre_v
            param_layer += 1
        
    model.load_state_dict(model_dict)
    print('HED init weight by model in: %s '%premodel_filename)
    return model


def main():
    model = HED(3)


if __name__ == '__main__':
    main()    


