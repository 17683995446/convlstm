import torch
import torch.nn as nn
import random
from model.sa_convlstm.DLinear_SAConvLSTM import SAConvLSTMCell
from model.convlstm.ConvLSTM import ConvLSTMCell
from  torch.optim.lr_scheduler import ReduceLROnPlateau
import math

class My_activation(nn.Module):

    def __init__(self):
        super(My_activation, self).__init__()
    #         self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        original_x = x
        # e = math.exp(1)
        # alpha_1 = -2.0
        # alpha_2 = -3.5
        # # print(torch.pow(e,x))
        # e_pow_x = torch.pow(e,x)
        # temp_1 = torch.pow(e_pow_x,alpha_1)
        # temp_2 = torch.pow(e_pow_x,alpha_2)
        # # x = (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
        # x = (1.0+temp_1)/(1.0+temp_2)
        # x = x*original_x
        x = torch.sin(16*x)
        x = x/8
        x += original_x
        return x
        # return x * torch.sigmoid(self.beta * x)
#         value = torch.tanh(self.beta)
#         return torch.tanh(x)#torch.relu(x)



# class SELayer(nn.Module):
#     def __init__(self, channel=64, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class Encode2Decode(nn.Module):
    """自回归,前t时刻预测后t时刻;无法做到语言翻译的效果,输入输出shape相同"""

    # self-attention convlstm for spatiotemporal prediction model
    def __init__(self, input_dim, hidden_dim, attn_hidden_dim, kernel_size, img_size=(16, 16), num_layers=4,
                 batch_first=True,
                 bias=True,
                 ):
        super(Encode2Decode, self).__init__()
        # self.generator = generator
        # self.discriminator = discriminator
        # self.optimizer_D = optimizer_D
        # self.optimizer_G = optimizer_G
        # self.scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=5, verbose=True)
        # self.scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=5, verbose=True)
        self.criterion = nn.MSELoss()
        self.img_size = img_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        #         self.highway = nn.Linear(10, 10)
        #         self.se_layer = SELayer()
        #         self.activation = nn.LeakyReLU(0.1)

        # encode:降低图片分辨率
        self.img_encode = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, (1, 1), (1, 1)),
            # My_activation(),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1)),
            # My_activation(),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1)),
            # My_activation(),
            nn.LeakyReLU(0.1)
        )
        # encode:还原图片分辨率
        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1), output_padding=(1, 1)),
            #             nn.LeakyReLU(0.1),
            # My_activation(),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1), output_padding=(1, 1)),
            #             nn.LeakyReLU(0.1),
            # My_activation(),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, input_dim, (1, 1), (1, 1)),
        )
        #Mogrifiers
        q_list,k_list,v_list=[],[],[]
        for i in range(20):
            q_list.append(nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, (1, 1))))
            k_list.append(nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, (1, 1))))
            v_list.append(nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, (1, 1))))
        self.layer_q = nn.ModuleList(q_list)
        self.layer_k = nn.ModuleList(k_list)
        self.layer_v = nn.ModuleList(v_list)


        cell_list, bns = [], []
        for i in range(0, self.num_layers):
            # cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cur_input_dim = self.hidden_dim
            cell_list.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            att_hidden_dim=self.attn_hidden_dim,
                                            bias=self.bias,
                                            ))
            # Use layer norm
            bns.append(nn.LayerNorm(normalized_shape=[hidden_dim, *self.img_size]))

        self.cell_list = nn.ModuleList(cell_list)
        self.bns = nn.ModuleList(bns)

        # Linear
    #         self.decoder_predict = nn.Conv2d(in_channels=hidden_dim,
    #                                          out_channels=input_dim,
    #                                          kernel_size=(1, 1),
    #                                          padding=(0, 0))

    def forward(self,ite,pre_epoch, x, y,generator,discriminator,optimizer_D,optimizer_G, teacher_forcing_rate=0.5, hidden_state=None,val_flag=False):
        if not self.batch_first:
            # (t,b,c,h,w)->(b,t,c,h,w)
            x = x.permute(1, 0, 2, 3, 4)
            y = y.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = x.shape
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h // 4, w // 4),device=x.device)
        seq_len, horizon = x.size(1), y.size(1)
        predict_temp_de = []
        predict_temp_de_0 = []
        frames = torch.cat([x, y], dim=1)
        train_losses_D = []
        train_losses_G = []

        for t in range(seq_len+horizon-1):

            if t < seq_len or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out
            # print("t,Xx_shape:",t,x.cpu().data.numpy().shape)
            x = self.img_encode(x)
            #             print("x.shape",x.cpu().data.numpy().shape)
            #             x_se = self.se_layer(x)
            #             print("t,X_reshape:",t,x.cpu().data.numpy().shape)
            #             x = self.se_layer(x)
            '''
            for i in range(1,self.mog_iterations+1):
        if (i % 2 == 0):
          ht = (2*torch.sigmoid(xt @ self.R)) * ht
        else:
          xt = (2*torch.sigmoid(ht @ self.Q)) * xt
      return xt, ht

            
            '''

            out = x
            for i, cell in enumerate(self.cell_list):
                if t>=1:
                    h_next, c_next, m_next, long_c_next = hidden_state[i]
                    # h_next, m_next = hidden_state[i]
                    if (t % 2 == 0):
                        x_t = out
                        y_t = h_next
                    else:
                        x_t = h_next
                        y_t = out
                    q=self.layer_q[t](x_t)
                    q = q.view(b, self.hidden_dim, h//4 * w//4)
                    q = q.transpose(1, 2)
                    k=self.layer_k[t](x_t)
                    k = k.view(b, self.hidden_dim, h//4 * w//4)
                    q_k = torch.bmm(q,k)
                    q_k = torch.softmax(q_k,dim=-1)
                    v=self.layer_v[t](x_t)
                    v = v.view(b, self.hidden_dim, h//4 * w//4)
                    x_t = torch.matmul(q_k,v.permute(0,2,1))
                    x_t = x_t.transpose(1, 2).view(b, self.hidden_dim, h//4, w//4)
                    y_t = (2*torch.sigmoid(x_t)) * y_t
                    if (t % 2 == 0):
                        h_next = y_t
                    else:
                        out = y_t

                    hidden_state[i] = (h_next, c_next, m_next, long_c_next)
                h_next, c_next, m_next, long_c_next = cell(out, hidden_state[i])
                out, hidden_state[i] = h_next, (h_next, c_next, m_next, long_c_next)
                out = self.bns[i](out)





            out = self.img_decode(out)


            #loss_1:
            # if t<seq_len:
            #     k = t+10
            # else:
            #     k = t

            if t>=seq_len-1:
                predict_temp_de.append(out)
                predict_temp_de_0.append(out)
            if val_flag:
                continue


            loss = self.criterion(out,frames[:, t+1, :, :, :])
            loss.backward(retain_graph=True)


            if ite>=pre_epoch:
                generator.train()
                discriminator.train()
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                #discriminator:
                for m,n in discriminator.named_parameters():
                    n.requires_grad=True

                D_Y = discriminator(frames[:, t+1, :, :, :])
                D_P_D = discriminator(generator(out.detach()).detach())
                # print(D_Y)
                # print(D_P_D)
                D_MSE = self.criterion(D_Y,D_P_D)

                D_Y = torch.mean(D_Y)
                D_P_D = torch.mean(D_P_D)
                loss_D = -D_Y+D_P_D
                loss_D.backward()
                optimizer_D.step()
                train_losses_D.append(loss_D.item())
                # print("lossD",t)


                #generator:

                with torch.autograd.set_detect_anomaly(True):

                    # for m,n in self.discriminator.named_parameters():
                    #     n.requires_grad=False
                    D_P_G = discriminator(generator(out.detach()))
                    D_Y = discriminator(frames[:, t+1, :, :, :])
                    G_MSE = self.criterion(D_P_G,D_Y)
                    _MSE = self.criterion(frames[:, t+1, :, :, :],generator(out.detach()))
                    D_P_G = torch.mean(D_P_G)
                    loss_G = -1/torch.log(1+_MSE) - 1/torch.log(1+G_MSE)
                    loss_G.backward()

                    optimizer_G.step()
                    train_losses_G.append(loss_G)

                    # if t==seq_len+horizon-1:

                    #     # loss_G = torch.mean(torch.stack(train_losses_G),dim=0)
                    #     loss_G.backward(retain_graph=True)
                    #     self.optimizer_G.step()



                # #loss_2:
                # loss = self.criterion(out,frames[:, t, :, :, :])
                # loss.backward(retain_graph=True)

            # else:
            #     predict_temp_de_0.append(out)
            # if t>=seq_len-1:
            #   predict_temp_de.append(out)
            #   predict_temp_de_0.append(out)

        if ite>=pre_epoch:
            for i in range(len(predict_temp_de)):
                predict_temp_de[i] = generator(predict_temp_de[i])
        predict_temp_de = torch.stack(predict_temp_de, dim=1)
        predict_temp_de = predict_temp_de[:, :, :, :, :]

        predict_temp_de_0 = torch.stack(predict_temp_de_0, dim=1)
        predict_temp_de_0 = predict_temp_de_0[:, :, :, :, :]

        #         x_in = frames[:, :seq_len, :, :, :]
        # #         print("x_in.shape",x_in.cpu().data.numpy().shape)
        #         z = x_in.permute(0,2,3,4,1).contiguous().view(-1,seq_len)
        # #         print("z1.shape",z.cpu().data.numpy().shape)
        #         z = self.highway(z)
        # #         print("z2.shape",z.cpu().data.numpy().shape)
        #         z = z.view(-1,1,64,64,10)
        # #         print("z3.shape",z.cpu().data.numpy().shape)
        #         z = z.permute(0,4,1,2,3)
        # #         print("z4.shape",z.cpu().data.numpy().shape)
        #         predict_temp_de = torch.mean(torch.cat([predict_temp_de.unsqueeze(0),z.unsqueeze(0)],dim=0),dim=0)
        import numpy as np
        if not val_flag and i>=pre_epoch:
            print("train_losses_D:",np.mean([k for k in train_losses_D]))
            print("train_losses_G",np.mean([k.item() for k in train_losses_G]))


        return predict_temp_de,predict_temp_de_0,generator,discriminator,optimizer_D,optimizer_G

    def _init_hidden(self, batch_size, image_size,device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size,device))
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
