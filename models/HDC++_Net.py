import torch
import torch.nn as nn
from torch.nn import functional as F
from .sync_batchnorm import SynchronizedBatchNorm3d


class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, padding, activation, stride=1):
        super(Conv_3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        SynchronizedBatchNorm3d(out_dim),
        activation)


class HDC_block(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(HDC_block, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4

        self.conv_3x3x1_left = Conv_3x3(self.out_inter_dim, self.out_inter_dim, (3, 3, 1), (1, 1, 0), activation)
        self.conv_1x3x3_left = Conv_3x3(self.out_inter_dim, self.out_inter_dim, (1, 3, 3), (0, 1, 1), activation)
        self.conv_3x1x3_left = Conv_3x3(self.out_inter_dim, self.out_inter_dim, (3, 1, 3), (1, 0, 1), activation)

        self.conv_3x3x1_right = Conv_3x3(self.out_inter_dim, self.out_inter_dim, (3, 3, 1), (1, 1, 0), activation)
        self.conv_1x3x3_right = Conv_3x3(self.out_inter_dim, self.out_inter_dim, (1, 3, 3), (0, 1, 1), activation)
        self.conv_3x1x3_right = Conv_3x3(self.out_inter_dim, self.out_inter_dim, (3, 1, 3), (1, 0, 1), activation)

        self.conv_1x1x1_left = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_right = Conv_1x1x1(out_dim, out_dim, activation)

        self.conv_1x1x1_top_left = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)

    def forward(self, x):

        x_1 = self.conv_1x1x1_left(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]

        x2 = self.conv_3x3x1_left(x2)
        x3 = self.conv_1x3x3_left(x2 + x3)
        x4 = self.conv_3x1x3_left(x3 + x4)

        x_top = torch.cat((x1, x2, x3, x4), dim=1)
        x_top = self.conv_1x1x1_top_left(x_top)
        x1_2 = x_top[:, 0:self.out_inter_dim, ...]
        x2_2 = x_top[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3_2 = x_top[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4_2 = x_top[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]

        x11 = x1 + x1_2
        x22 = self.conv_3x3x1_right(x2 + x2_2)
        x33 = self.conv_1x3x3_right(x3 + x3_2)
        x44 = self.conv_3x1x3_right(x4 + x4_2)

        xx = torch.cat((x11, x22, x33, x44), dim=1)
        xx = self.conv_1x1x1_right(xx)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        return xx + x


device1 = torch.device("cuda")
def hdc(image, num=2):
    x1 = torch.Tensor([]).to(device1)
    for i in range(num):
        for j in range(num):
            for k in range(num):
                x3 = image[:, :, k::num, i::num, j::num]
                x1 = torch.cat((x1, x3), dim=1)
    return x1


class HDC_Net(nn.Module):

    def __init__(self, in_dim, out_dim, num_filters):
        super(HDC_Net, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)

        self.down = Conv_3x3(self.in_dim, self.n_f, 3, 1, self.activation, stride=2)
        # down
        self.conv_1 = HDC_block(self.n_f, self.n_f, self.activation)
        self.down_1 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_2 = HDC_block(self.n_f, self.n_f, self.activation)
        self.down_2 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_3 = HDC_block(self.n_f, self.n_f, self.activation)
        self.down_3 = Conv_down(self.n_f, self.n_f, self.activation)
        # bridge
        self.bridge = HDC_block(self.n_f, self.n_f, self.activation)
        # up
        self.up_1 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_4 = HDC_block(self.n_f * 2, self.n_f, self.activation)
        self.up_2 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_5 = HDC_block(self.n_f * 2, self.n_f, self.activation)
        self.up_3 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_6 = HDC_block(self.n_f * 2, self.n_f, self.activation)
        self.up = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.out = nn.Conv3d(self.n_f, out_dim, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  
            elif isinstance(m, SynchronizedBatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.down(x)
        x1 = self.conv_1(x)
        x = self.down_1(x1)
        x2 = self.conv_2(x)
        x = self.down_2(x2)
        x3 = self.conv_3(x)
        x = self.down_3(x3)
        x = self.bridge(x)
        x = self.up_1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.conv_4(x)
        x = self.up_2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.conv_5(x)
        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_6(x)
        x = self.up(x)
        x = self.out(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    # from thop import profile
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 4, 112, 112, image_size), device=device)
    print("x size: {}".format(x.size()))
    model = HDC_Net(in_dim=4, out_dim=4, num_filters=32).to(device)
    # flops, params = profile(model, inputs=(x,))
    # print("***********")
    # print(flops, params)
    # print("***********")
    out = model(x)
    print("out size: {}".format(out.size()))
