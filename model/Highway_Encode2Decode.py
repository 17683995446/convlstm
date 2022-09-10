import torch
import torch.nn as nn
import random
# from model.sa_convlstm.Highway_SAConvLSTM import SAConvLSTMCell
from model.convlstm.ConvLSTM import ConvLSTMCell


class My_activation(nn.Module):

    def __init__(self):
        super(My_activation, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # return x * torch.sigmoid(self.beta * x)

        print("x_encode:",x.shape)
        if x.shape[-1] == 16:
            for i in range(9):
                for j in range(16):
                    for k in range(16):
                        print(x.cpu().data.numpy()[0][0][i][j][k],end=" ")
                    print(i,j)
        import time
        # time.sleep(1)
        # import matplotlib.pyplot as plt
        # for l in range(9):
        #     plt.subplot(1,9,l+1)
        #     plt.imshow(x.cpu().data.numpy()[0][0][l])
        #     # plt.subplot(2,9,l+28)
        #     # plt.imshow(x.cpu().data.numpy()[0][0][l+9])
        # plt.show()
        # value = torch.sigmoid(self.beta)
        import math
        e = math.exp(1)
        alpha = -4.0
        # print(torch.pow(e,x))
        # x = (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
        x = (1.0-torch.pow(e,alpha*x))/(1.0+torch.pow(e,alpha*x))
        return x#torch.relu(x)#torch.relu(x)




class Encode2Decode(nn.Module):
    """自回归,前t时刻预测后t时刻;无法做到语言翻译的效果,输入输出shape相同"""

    # self-attention convlstm for spatiotemporal prediction model
    def __init__(self, input_dim, hidden_dim, attn_hidden_dim, kernel_size, img_size=(16, 16), num_layers=4,
                 batch_first=True,
                 bias=True,
                 ):
        super(Encode2Decode, self).__init__()
        self.img_size = img_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.act = My_activation()

        # encode:降低图片分辨率
        self.img_encode = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, (1, 1,1), (1, 1,1)),
            # nn.LeakyReLU(0.1),
            My_activation(),
            nn.LeakyReLU(0.1),
            nn.Conv3d(hidden_dim, hidden_dim, (3, 3,3), (1, 2,2), (1, 1,1),groups=hidden_dim),
            My_activation(),
            nn.LeakyReLU(0.1),
            # nn.Conv3d(hidden_dim, hidden_dim, (3, 3,3), (2, 2,1), (1, 1,1)),
            # nn.LeakyReLU(0.1)
        )
        # encode:还原图片分辨率
        self.img_decode = nn.Sequential(
            # nn.ConvTranspose3d(hidden_dim, hidden_dim, (3, 3,3), (2, 2,1), (1, 1,1), output_padding=(1, 1,0)),
            # nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, (3, 3,3), (1, 2,2), (1, 1,1), output_padding=(0, 1,1)),
            My_activation(),
            nn.LeakyReLU(0.1),
            nn.Conv3d(hidden_dim, input_dim, (1, 1,1), (1, 1,1)),
        )
        cell_list, bns = [], []
        for i in range(0, self.num_layers):
            # cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cur_input_dim = self.hidden_dim
            # cell_list.append(SAConvLSTMCell(input_dim=cur_input_dim,
            #                                 hidden_dim=self.hidden_dim,
            #                                 kernel_size=self.kernel_size,
            #                                 att_hidden_dim=self.attn_hidden_dim,
            #                                 bias=self.bias,
            #                                 ))
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          bias=self.bias,
                                          ))
            # Use layer norm
            bns.append(nn.LayerNorm(normalized_shape=[hidden_dim, *self.img_size]))

        self.cell_list = nn.ModuleList(cell_list)
        self.bns = nn.ModuleList(bns)

        # Linear
        self.decoder_predict = nn.Conv2d(in_channels=hidden_dim,
                                         out_channels=input_dim,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden_state=None):
        print(x.shape)
        if not self.batch_first:
            # (t,b,c,h,w)->(b,t,c,h,w)
            x = x.permute(1, 0, 2, 3, 4,5)
            y = y.permute(1, 0, 2, 3, 4,5)
        b, _, _,p, h, w = x.shape
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(p,h // 2, w // 2),device=x.device)
        seq_len, horizon = x.size(1), y.size(1)
        predict_temp_de = []
        frames = torch.cat([x, y], dim=1)
        # out_list1 = []
        # out_list2 = []
        # out_list3 = []
        for t in range(seq_len+horizon-1):
            if t < seq_len or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :,:]
            else:
                x = out
            # print(x[0][0])
            print("x:",x.cpu().data.numpy().shape)
            import matplotlib.pyplot as plt
            for l in range(9):
                plt.subplot(4,9,l+1)
                plt.imshow(x.cpu().data.numpy()[0][0][l])
                plt.subplot(4,9,l+10)
                plt.imshow(x.cpu().data.numpy()[0][0][l+9])

            if x.shape[-1] == 16:
                for i in range(9):
                    for j in range(16):
                        for k in range(16):
                            print(x.cpu().data.numpy()[0][0][i][j][k],end=" ")
                        print(i,j)


            x = self.img_encode(x)
            import matplotlib.pyplot as plt
            for l in range(9):
                plt.subplot(4,9,l+19)
                plt.imshow(x.cpu().data.numpy()[0][0][l])
                plt.subplot(4,9,l+28)
                plt.imshow(x.cpu().data.numpy()[0][0][l+9])
            # plt.show()
            plt.savefig("./outputs/train.png")
            # print("x:",x.cpu().data.numpy().shape)
            out = x
            for i, cell in enumerate(self.cell_list):
                h_next, c_next = cell(out, hidden_state[i])
                out, hidden_state[i] = h_next, (h_next, c_next)
                out = self.bns[i](out)
                # if i==0:
                #     h_next, c_next, m_next = cell(x, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list1.append(out)
                # if i==1 and t%2==1:
                #     h_next, c_next, m_next = cell(x, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list2.append(out)
                #     h_next, c_next, m_next = cell(out, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list2.append(out)
                #     # print("out:",out.cpu().data.numpy().shape)
                # if i==2 and t%5==4:
                #     h_next, c_next, m_next = cell(x, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list3.append(out)
                #     h_next, c_next, m_next = cell(out, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list3.append(out)
                #     h_next, c_next, m_next = cell(out, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list3.append(out)
                #     h_next, c_next, m_next = cell(out, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list3.append(out)
                #     h_next, c_next, m_next = cell(out, hidden_state[i])
                #     out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                #     out = self.bns[i](out)
                #     out_list3.append(out)


            out = self.img_decode(out)
            predict_temp_de.append(out)
        # print(len(out_list1),len(out_list2),len(out_list3))
        # for i in range(seq_len):
        #     temp=out_list1[i]+out_list2[i]+out_list3[i]
        #     predict_temp_de.append(self.img_decode(temp))
        #     out = self.img_decode(out)

        predict_temp_de = torch.stack(predict_temp_de, dim=1)
        predict_temp_de = predict_temp_de[:, seq_len-1:, :, :, :]
        return predict_temp_de

    def _init_hidden(self, batch_size, image_size,device):
        init_states = []
        for i in range(self.num_layers):
            # init_states.append(self.cell_list[i].init_hidden(batch_size, image_size,device))
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))

        return init_states


def main():
    model = Encode2Decode(1, 16, 16, (3, 3))
    img_size = 1, 64, 64
    batch_size, seq_len, horizon = 2, 10, 10
    x = torch.rand(batch_size, seq_len, *img_size)
    y = torch.rand(batch_size, horizon, *img_size)
    y_hat = model(x, y)
    print(y_hat.shape)


if __name__ == '__main__':
    main()
