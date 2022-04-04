
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from settings import run_folder, run_archive_folder
class Residual_CNN(nn.Module):
    def __init__(self, reg_const, input_dim, output_dim, hidden_layers):
      
        super(Residual_CNN , self).__init__()
        # filter(out_channel), kernel_size는 config의 hidden layer 그대로 복사
        # filter = 75 , kernel = (4,4)
        # padding = same -> 1         -> this was solved since Pytorch 1.10.0 “same” keyword is accepted as input for padding for conv2d
        # x = add([input_block, x])   ( https://tensorflow.google.cn/api_docs/python/tf/keras/layers/add  )    -> torch.add https://pytorch.org/docs/stable/generated/torch.add.html
        # l2 regularizer는 loss파트에서 weight decay를 조정  https://discuss.pytorch.org/t/how-to-implement-pytorch-equivalent-of-keras-kernel-weight-regulariser/99773
     
        self.reg_const=reg_const
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_layer=hidden_layers
        # print("input_dim : ",self.input_dim)
        # print("output_dim : ",self.output_dim)
        # input_dim은 (2,6,7)이므로, 첫 입력에 첫 채널이 들어간다고 생각.
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 75, kernel_size=4, padding="same"),
            nn.BatchNorm2d(75),
            nn.LeakyReLU()
        )
        
        self.conv_and_residual_layer = nn.Sequential(
            # conv_layer
            nn.Conv2d(75, 75, kernel_size=4, padding="same"),
            nn.BatchNorm2d(75),
            nn.LeakyReLU(),

            # residual_layer
            nn.Conv2d(75, 75, kernel_size=4, padding="same"),
            nn.BatchNorm2d(75)
        )

        self.value_head1 = nn.Sequential(
            nn.Conv2d(75, 1, kernel_size=1, padding="same"),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.value_head2 = nn.Sequential(
            nn.Linear(42, 20),                     #<<
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        self.policy_head1 = nn.Sequential(
            nn.Conv2d(75, 1, kernel_size=1, padding="same"),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.policy_head2 = nn.Sequential(
            nn.Linear(42, self.output_dim)         #<<
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        input_block = x

        for i in range(5):
            x = self.conv_and_residual_layer(x)
            x = torch.add(input_block,x)
            x = F.leaky_relu(x)


        vh = self.value_head1(x)
        vh = vh.view(vh.size(0), -1)

        # print("vh : ",vh.shape)

        vh = self.value_head2(vh)

        ph = self.policy_head1(x)
        ph = ph.view(ph.size(0), -1)

        # print("ph : ",ph.shape)

        ph = self.policy_head2(ph)

        return {'value_head':vh,'policy_head':ph}

    def convertToModelInput(self, state):
        inputToModel = state.binary  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
        inputToModel = np.reshape(inputToModel, self.input_dim)
        return (inputToModel)

    def write(self,game,version):
        torch.save(self.model.state_dict(), (run_folder + 'models/version' + "{0:0>4}".format(version) + '.pt'))
    
    def read(self,game,run_number,version):
        return torch.load(run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + 'pt')