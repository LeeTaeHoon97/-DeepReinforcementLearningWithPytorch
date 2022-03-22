# %matplotlib inline

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_CNN(nn.Module):
    def __init__(self, reg_const, input_dim, output_dim, hidden_layers):
        super(Residual_CNN, self).__init__()
        # filter(out_channel), kernel_size는 config의 hidden layer 그대로 복사
        # filter = 75 , kernel = (4,4)
        # padding = same -> 1         -> this was solved since Pytorch 1.10.0 “same” keyword is accepted as input for padding for conv2d
        # x = add([input_block, x])   ( https://tensorflow.google.cn/api_docs/python/tf/keras/layers/add  )    -> torch.sum https://pytorch.org/docs/stable/generated/torch.sum.html
        # l2 regularizer는 loss파트에서 weight decay를 조정  https://discuss.pytorch.org/t/how-to-implement-pytorch-equivalent-of-keras-kernel-weight-regulariser/99773
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(self.input_dim, 75, kernel_size=4, padding="same"),
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
            nn.Linear("vh.size(0)", 20)
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
            nn.Linear("ph.size(0)", output_dim)
        )

    def forward(self, x):
        input_block = x
        x = self.hidden_layer(x)

        for i in range(5):
            x = self.conv_and_residual_layer(x)
            x = torch.sum([input_block, x], keepdim=True)
            x = F.leaky_relu(x)

        vh = self.value_head1(x)
        vh = vh.view(vh.size(0), -1)
        vh = self.value_head2(vh)

        ph = self.policy_head1(x)
        ph = ph.view(ph.size(0), -1)
        ph = self.policy_head2(ph)

        return x