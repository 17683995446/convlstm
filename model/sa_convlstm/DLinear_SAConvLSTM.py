import torch
import torch.nn as nn


class SA_Attn_Mem(nn.Module):
    # SAM 自注意力模块
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, (1, 1))

        self.layer_v = nn.Conv2d(input_dim, input_dim, (1, 1))
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, (1, 1))

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, (1, 1))
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, (1, 1))

    def forward(self, h, m ):

        batch_size, channels, H, W = h.shape
        # **********************  feature aggregation ******************** #

        # Use 1x1 convolution for Q,K,V Generation
        K_h = self.layer_k(h)
        # print("start___K_h.shape",K_h.cpu().data.numpy().shape)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)

        Q_h = self.layer_q(h)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)

        K_m = self.layer_k2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)

        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)

        # print(V_h.cpu().data.numpy()[0])

        # print("K_h.shape",K_h.cpu().data.numpy().shape)
        # print("Q_h.shape",Q_h.cpu().data.numpy().shape)
        # print("V_h.shape",V_h.cpu().data.numpy().shape)
        # print("K_h.shape",K_h.cpu().data.numpy().shape)

        # **********************  hidden h attention ******************** #
        # [batch_size,H*W,H*W]

        Q_K_h = torch.bmm(Q_h, K_h)




        A_h = torch.softmax(Q_K_h, dim=-1)
        # print("A_h.shape",A_h.cpu().data.numpy().shape)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        # print("Z_h1.shape",Z_h.cpu().data.numpy().shape)
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        # print("Z_h2.shape",Z_h.cpu().data.numpy().shape)
        # **********************  memory m attention ******************** #
        # [batch_size,H*W,H*W]
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)

        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        # print("Zm.shape:",Z_m.cpu().data.numpy().shape)
        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)   # [batch_size,in_channels*2,H,W]


        # q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # size=[B, H, E, L]
        # k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # res = q_fft * torch.conj(k_fft)
        # corr = torch.fft.irfft(res, dim=-1) # size=[B, H, E, L]

        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        #
        mi = torch.sigmoid(mi)

        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        # avg_m = torch.cat([res.unsqueeze(0),h.unsqueeze(0),new_h.unsqueeze(0)],dim=0)
        # print("temp_m_pre.shape",avg_m.cpu().data.numpy().shape)
        # avg_m = torch.min(avg_m,dim=0).values
        # print("avg_m.shape",avg_m.cpu().data.numpy().shape)
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(avg_m.cpu().data.numpy()[0][0])
        # plt.subplot(122)
        # plt.imshow(m.cpu().data.numpy()[0][0])
        # plt.show()
        # res_m = m-avg_m
        # print("new_res.shape",new_res.cpu().data.numpy().shape)
        # print("temp_m.shape",avg_m.cpu().data.numpy().shape)
        # new_m = new_m+res_m

        return new_h, new_m


class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, att_hidden_dim, kernel_size, bias):
        """
           Initialize SA ConvLSTM cell.
           Parameters
           ---------
           input_dim: int
               Number of channels of input tensor.
           hidden_dim: int
               Number of channels of hidden state.
           kernel_size: (int, int)
               Size of the convolutional kernel.
           bias: bool
               Whether to add the bias.
           att_hidden_dim: int
               Number of channels of attention hidden state
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.attention_layer = SA_Attn_Mem(hidden_dim, att_hidden_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim)
        )

        self.conv_long = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim)
        )

    def forward(self, input_tensor, cur_state):
        # print("input_tensor",input_tensor.cpu().data.numpy().shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(input_tensor.cpu().data.numpy()[0][0])
        # plt.show()


        h_cur, c_cur, m_cur ,long_c_cur = cur_state
        # print("h_cur:",h_cur.cpu().data.numpy().shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        # ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇这里是新插入的内容，为了改编LSTM结构，如果没用，最后要删除这部分。
        combined_long = torch.cat([h_cur, c_next], dim=1)
        combined_conv_long = self.conv_long(combined_long)
        long_cc_i, long_cc_f, long_cc_o, long_cc_g = torch.split(combined_conv_long, self.hidden_dim, dim=1)
        long_i = torch.sigmoid(long_cc_i)
        long_f = torch.sigmoid(long_cc_f)
        long_o = torch.sigmoid(long_cc_o)
        long_g = torch.tanh(long_cc_g)

        long_c_next = long_f * long_c_cur + long_i * long_g
        c_next = long_o * torch.tanh(long_c_next)
        # ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆这里是新插入的内容，为了改编LSTM结构，如果没用，最后要删除这部分。
        h_next = o * torch.tanh(c_next)

        h_next, m_next  = self.attention_layer(h_next, m_cur)
        return h_next, c_next, m_next ,long_c_next

    # initialize h, c, m
    def init_hidden(self, batch_size, image_size,device):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        m = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        res = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        return h, c, m, res


class SAConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_hidden_dim, kernel_size, num_layers, batch_first=False, bias=True,
                 return_all_layers=False):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        # make sure that both "kernel_size" and 'hidden_dim' are lists having len=num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        attn_hidden_dim = self._extend_for_multilayer(attn_hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(attn_hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            att_hidden_dim=self.attn_hidden_dim[i],
                                            bias=self.bias,
                                            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t,b,c,h,w)->(b,t,c,h,w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w),device=input_tensor.device)
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c, m ,long_c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c, m ,long_c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c, m ,long_c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c, m ,long_c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size,device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size,device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
                isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    x = torch.rand((32, 10, 64, 16, 16))
    # input_dim, hidden_dim, kernel_size, num_layers, batch_first = False, bias = True,return_all_layers = False
    sa_convlstm = SAConvLSTM(64, 16, 16, (3, 3), 2, True, True, True)
    _, last_states = sa_convlstm(x)
    h = last_states[0][0]  # 0 for layer index, 0 for h index
    # print(h.shape)
