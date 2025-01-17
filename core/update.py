import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import ResidualBlock
DEBUG = False


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)  # correlation+flow
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.sample_num * args.corr_levels  # 27
        self.convc1 = nn.Conv2d(cor_planes, 64, 3, padding=1)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convc3 = nn.Conv2d(64, 48, 3, padding=1)
        self.convf1 = nn.Conv2d(3 * args.gauss_num, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64 - 3 * args.gauss_num, 3, padding=1)

    def forward(self, disp, corr, w, sigma):
        N, C, H, W = corr.shape
        corr = corr.reshape(N, self.args.corr_levels, self.args.gauss_num, self.args.sample_num, H, W).permute(0, 2, 1,
                                                                                                               3, 4, 5)
        corr = corr.reshape(-1, self.args.corr_levels * self.args.sample_num, H, W)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = F.relu(self.convc3(cor))
        cor = cor.reshape(N, -1, H, W)
        param = torch.cat((disp, w.detach(), sigma.detach()), dim=1)
        param_f = F.relu(self.convf1(param))
        param_f = F.relu(self.convf2(param_f))
        return torch.cat([cor, param_f, param], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class ParametersUpdater(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(ParametersUpdater, self).__init__()
        self.args = args
        self.head = FlowHead(input_dim, hidden_dim, args.gauss_num)
        self.sigma0 = 0.5
        self.eps = 1e-3
        self.gamma1 = 1
        self.gamma2 = 1
        self.gamma3 = 1

    def forward(self, hidden_state, mu, sigma, w):
        delta = self.head(hidden_state)
        _, M, _, _ = delta.shape

        # feed forward gradients
        delta_sigma = 0.5 * (((1 - M * w) * sigma ** 2 - self.sigma0 ** 2 - delta ** 2) / (M * sigma ** 3) + w * sigma / (self.sigma0 ** 2))
        delta_mu = -0.5 * delta * (1 / (M * sigma ** 2) + w / (self.sigma0 ** 2))
        beta = 0.5 * (-1 / (M * w + self.eps) + torch.log(self.sigma0 * M * w / sigma + self.eps) + (sigma ** 2 + delta ** 2) / (2 * self.sigma0 ** 2) + 0.5)
        delta_w = beta - torch.sum(beta, dim=1, keepdim=True) / M

        # clip the gradients
        delta_sigma = torch.clip(delta_sigma * self.gamma1, min=-3, max=3)
        delta_mu = torch.clip(delta_mu * self.gamma2, min=-128, max=128)
        delta_w = torch.clip(delta_w * self.gamma3, min=-1 / (M * 4), max=1 / (M * 4))

        # update & clip the parameters
        sigma = torch.clip(sigma - delta_sigma, min=0.1, max=16)
        mu = mu - delta_mu
        w = torch.clip(w - delta_w, min=0, max=1)
        # normalize
        w = w / torch.sum(w, dim=1, keepdim=True)
        return mu, w, sigma


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=None):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 256
        self.gru04 = ConvGRU(hidden_dims[3], encoder_output_dim + hidden_dims[2] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[2], 128 + hidden_dims[1] * (args.n_gru_layers > 2) + hidden_dims[3])
        self.gru16 = ConvGRU(hidden_dims[1], 128 + hidden_dims[0] * (args.n_gru_layers > 3) + hidden_dims[2])
        factor = 2 ** self.args.n_downsample
        self.mask = nn.Sequential(nn.Conv2d(hidden_dims[3], 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))
        self.ParametersUpdater = ParametersUpdater(self.args, input_dim=128, hidden_dim=256)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU())
        self.conv2_out = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.conv3_out = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())

    def forward(self, net, inp, corr=None, mu=None, w=None, sigma=None, iter04=True, iter08=True, iter16=True,
                update=True, test_mode=False, motion_features_list=None):
        if motion_features_list is None:
            if self.args.n_gru_layers >= 1:
                motion_features = self.encoder(mu, corr, w, sigma)
                motion_features_list = [motion_features]
            if self.args.n_gru_layers >= 2:
                motion_features_08_0 = self.conv2(motion_features.detach())
                motion_features_08 = self.conv2_out(motion_features_08_0)
                motion_features_list = [motion_features, motion_features_08]
            if self.args.n_gru_layers >= 3:
                motion_features_16 = self.conv3(motion_features_08_0.detach())
                motion_features_16 = self.conv3_out(motion_features_16)
                motion_features_list = [motion_features, motion_features_08, motion_features_16]

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), motion_features_list[2], pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), motion_features_list[1], pool2x(net[0]),
                                    interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), motion_features_list[1], pool2x(net[0]))
        if iter04:
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features_list[0], interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features_list[0])

        if not update:
            return net, motion_features_list

        mu, w, sigma = self.ParametersUpdater(net[0], mu, sigma, w)

        if not test_mode:
            mask = self.mask(net[0]) * 0.25
        else:
            mask = torch.zeros_like(mu)
        return net, mask, mu, sigma, w


import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityDecoder(nn.Module):
    def __init__(self, max_disp, channels):
        super(DisparityDecoder, self).__init__()
        # self.params = params
        max_disp = int(max_disp/ (2**4))
        self.channel_multiplier = 4
        self.base_channels = [1, 2, 4, 8, 16, 32]
        l1, l2, l3, l4, l5 = channels

        f10, f11, f12, f13, f14, f15 = [
            x * self.channel_multiplier for x in self.base_channels
        ] # 4 8 16 32 64 128

        f20, f21, f22, f23, f24, f25 = [
            x * 4 * self.channel_multiplier for x in self.base_channels
        ] # 16 32 64 128 256 512

        self.attention1 = DisparityAttention(in_channels=l5*2, max_disparity=max_disp)
        self.layer5 = self._make_identity_layer(l5*2)
        self.resup1 = nn.Sequential(nn.ConvTranspose2d(l5*2+max_disp+1, f14, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f14),
                                    nn.ReLU(inplace=True))

        self.attention2 = DisparityAttention(in_channels=l4*2, max_disparity=3)
        self.layer4 = self._make_identity_layer(l4*2+f22)
        self.resup2 = nn.Sequential(nn.ConvTranspose2d(l4*2+f22+3+1, f14, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f14),
                                    nn.ReLU(inplace=True))
        
        self.attention3 = DisparityAttention(in_channels=l3*2, max_disparity=3)
        self.layer3 = self._make_identity_layer(l3*2+f22)
        self.resup3 = nn.Sequential(nn.ConvTranspose2d(l3*2+f22+3+1, f13, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f13),
                                    nn.ReLU(inplace=True))
        
        self.attention4 = DisparityAttention(in_channels=l2*2, max_disparity=3)
        self.layer2 = self._make_identity_layer(l2*2+f21)
        self.resup4 = nn.Sequential(nn.ConvTranspose2d(l2*2+f21+3+1, f12, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f12),
                                    nn.ReLU(inplace=True))
        
        self.attention5 = DisparityAttention(in_channels=l1*2, max_disparity=3)
        self.layer1 = self._make_identity_layer(l1*2+f21)
        self.resup5 = nn.Sequential(nn.ConvTranspose2d(l1*2+f21+3+1, f10, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f10),
                                    nn.ReLU(inplace=True))

        
        
        
        # self.layer2 = self._make_identity_layer(f23+f21)
        


        # self.attention1 = DisparityAttention(in_channels=f24, max_disparity=max_disp)
        
        # self.attention2 = DisparityAttention(in_channels=f24, max_disparity=3)
        
        # self.attention3 = DisparityAttention(in_channels=f23, max_disparity=3)
        
        # self.attention4 = DisparityAttention(in_channels=f23, max_disparity=3)
        
        # self.attention5 = DisparityAttention(in_channels=f12, max_disparity=3)
        self.resred = nn.Conv2d(
            l1*2+f12,
            f20,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        # self.resred4 = ResReduce(f20 // 2, name="resred_4")

        self.head1 = nn.Conv2d(
            f20+3,
            f20,  # or 128?
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.bn = nn.BatchNorm2d(f20)

        self.disp_head = nn.ModuleList([
            nn.Conv2d(
                x,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ) for x in [l5*2, l4*2+f22, l3*2+f22, l2*2+f21, l1*1]
        ])

    def _make_identity_layer(self, dim):
        layer1 = ResidualBlock(dim, dim, stride=1)
        layer2 = ResidualBlock(dim, dim, stride=1)
        layer3 = ResidualBlock(dim, dim, stride=1)

        layers = (layer1, layer2, layer3)

        # self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, inputs):
        f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = inputs[0]
        f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = inputs[1]
        if DEBUG:
            print("f_l_1.shape", f_l_1.shape)
            print("f_r_1.shape", f_r_1.shape)

        attention = []
        x, pw = self.attention1([f_l_5, f_r_5])
        attention.append(x)

        if DEBUG:
            print("x.shape after attention1", x.shape)


        x = self.layer5(x)

        if DEBUG:
            print('x.shape after layer5 ', x.shape)
        
        disp5 = self.disp_head[0](x)
        x = torch.cat((x, pw[0], disp5), dim=1)
        if DEBUG:
            print('cat x, pw, disp5', x.shape)
            print('pw', pw[0].shape)
            print('disp5', disp5.shape)
        x = self.resup1(x)
        x_skip = x

        if DEBUG:
            print("x.shape after resup1", x.shape)

        x, pw = self.attention2([f_l_4, f_r_4], pw)
        attention.append(x)
        x = torch.cat((x, x_skip), dim= 1)

        if DEBUG:
            print("x.shape after attention2", x.shape)

        x = self.layer4(x)
        if DEBUG:
            print('x.shape after layer4 ', x.shape)
        disp4 = self.disp_head[1](x)
        x = torch.cat((x, pw[1], disp4), dim=1)
        if DEBUG:
            print('cat x, pw, disp5', x.shape)
        x = self.resup2(x)
        x_skip = x

        if DEBUG:
            print("x.shape after resup2", x.shape)

        
        x, pw = self.attention3([f_l_3, f_r_3], pw)
        attention.append(x)
        x = torch.cat((x, x_skip), dim=1)

        if DEBUG:
            print("x.shape after attention3", x.shape)

        x = self.layer3(x)
        if DEBUG:
            print('x.shape after layer3 ', x.shape)
        disp3 = self.disp_head[2](x)

        if DEBUG:
            print("x.shape after layer3", x.shape)
        x = torch.cat((x, pw[2], disp3), dim=1)
        if DEBUG:
            print('cat x, pw, disp5', x.shape)
        x = self.resup3(x)
        x_skip = x

        if DEBUG:
            print("x.shape after resup3", x.shape)

        
        x, pw = self.attention4([f_l_2, f_r_2], pw)
        attention.append(x)
        x = torch.cat((x, x_skip), dim=1)

        if DEBUG:
            print("x.shape after attention4", x.shape)

        x = self.layer2(x)
        disp2 = self.disp_head[3](x)
        x = torch.cat((x, pw[3], disp2), dim=1)
        if DEBUG:
            print('cat x, pw, disp5', x.shape)
            print('disp5', disp2.shape)
            print('pw', pw[3].shape)
        x = self.resup4(x)

        if DEBUG:
            print("x.shape after resup4", x.shape)

        x_skip = x
        x, pw = self.attention5([f_l_1, f_r_1], pw)
        attention.append(x)
        x = torch.cat((x, x_skip), dim=1)

        x = self.resred(x)
        x = torch.cat((x, pw[4]), dim=1)
        x = self.head1(x)
        x = self.bn(x)
        stereo_disp = self.disp_head[4](x)

        if DEBUG:
            print("x.shape after head1", x.shape)

        return stereo_disp, disp2, disp3, disp4, disp5


class DisparityAttention(nn.Module):
    def __init__(self,in_channels, max_disparity=3):
        super(DisparityAttention, self).__init__()
        self.max_disparity = max_disparity
        # self.convs_block = self.create_conv_block(in_channels, in_channels, 3, 2)
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # Adjust based on input channels
            out_channels=max_disparity,
            kernel_size=(3, max_disparity + 2),
            padding="same"
        )

        self.softmax = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def create_conv_block(self, in_channels, out_channels, kernel_size, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,  # Use in_channels for the first layer
                out_channels,
                kernel_size,
                padding='same' # Same padding
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)



    def forward(self, inputs, previous_weights=None):
        # Assume inputs is a list [left_image, right_image]
        left_image, right_image = inputs
        if DEBUG:
            print('left_image shape', left_image.shape)
        
        # Roll the right image to the left by max_disparity//2 pixels if previous_weights is given
        if previous_weights is not None:
            shift = -self.max_disparity // 2
        else:
            shift = -1
        right_image = torch.roll(right_image, shifts=shift, dims=3)

        # Add previous weights to the new list of weights and to the right image
        new_previous_weights = []
        if previous_weights is not None:
            if DEBUG:
                print(len(previous_weights))
            n = len(previous_weights) + 1
            for j, pw in enumerate(previous_weights):
                if DEBUG:
                    print('pw',pw.shape)
                    print('ri', right_image.shape)
                pw = self.upsample(pw)
                if DEBUG:
                    print('pw after upsampling', pw.shape)
                new_previous_weights.append(pw)
                for i in range(self.max_disparity):
                    # Select the weights for disparity i and apply them to the right image shifted by i pixels
                    pw_right = pw[:,i:i+1,  :, :] * torch.roll(right_image, shifts=-i * (2 ** (n - j)), dims=3)
                    right_image += pw_right
                if DEBUG:
                    print('after applying previous weights', right_image.shape)

        # Stack left and right images
        stacked = torch.cat([left_image, right_image], dim=1)  # (B, C, H, W)
        # stacked = self.convs_block(stacked)
        if DEBUG:
            print('stacked', stacked.shape)

        # Compute disparity
        disparity_map = self.conv(stacked)  # (B, max_disparity, H, W)
        if DEBUG:
            print('disparity_map', disparity_map.shape)

        # Apply softmax to get the weights
        weights = self.softmax(disparity_map)
        if DEBUG:
            print('weights', weights.shape)
        new_previous_weights.append(weights)

        # Compute attended right image
        attended_right = torch.zeros_like(right_image)
        for i in range(self.max_disparity):
            # Select the weights for disparity i and apply them to the right image shifted by i pixels
            # weighted_right = weights[:, :, :, i:i+1] * torch.roll(right_image, shifts=-i, dims=2)
            weighted_right = weights[:, i:i+1, :, :] * torch.roll(right_image, shifts=-i, dims=3)
            attended_right += weighted_right
        if DEBUG:
            print('attended_right', attended_right.shape)

        # Concatenate left image and attended right image
        x = torch.cat([left_image, attended_right], dim=1)  # (B, C, H, W)
        if DEBUG:
            print('torch cat (left_image, attended_right)', x.shape)

        return x, new_previous_weights
    


class DisparityDecoderv2(nn.Module):
    def __init__(self):
        super(DisparityDecoderv2, self).__init__()
        # self.params = params
        self.channel_multiplier = 4
        self.base_channels = [1, 2, 4, 8, 16, 32]

        f10, f11, f12, f13, f14, f15 = [
            x * self.channel_multiplier for x in self.base_channels
        ]

        f20, f21, f22, f23, f24, f25 = [
            x * 4 * self.channel_multiplier for x in self.base_channels
        ]

        self.resup1 = nn.Sequential(nn.ConvTranspose2d(f24, f14, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f14),
                                    nn.ReLU(inplace=True))
        self.resup2 = nn.Sequential(nn.ConvTranspose2d(f24+f22, f14, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f14),
                                    nn.ReLU(inplace=True))
        self.resup3 = nn.Sequential(nn.ConvTranspose2d(f24+f22, f13, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f13),
                                    nn.ReLU(inplace=True))
        self.resup4 = nn.Sequential(nn.ConvTranspose2d(f23+f21, f12, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f12),
                                    nn.ReLU(inplace=True))
        self.resup5 = nn.Sequential(nn.ConvTranspose2d(f23+f21, f10, kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(f10),
                                    nn.ReLU(inplace=True))

        self.layer5 = self._make_identity_layer(f24)
        self.layer4 = self._make_identity_layer(f24+f22)
        self.layer3 = self._make_identity_layer(f24+f22)
        self.layer2 = self._make_identity_layer(f23+f21)
        self.layer1 = self._make_identity_layer(f23+f21)


        self.attention1 = DisparityAttentionv2(in_channels=f24, max_disparity=21)
        # self.resid1 = (filters=(f15, f25), name="resid_1_1")
        # self.resid2 = ResidualBlock()#ResIdentity(filters=(f15, f25), name="resid_1_2")
        # self.resid3 = ResidualBlock(filters=(f15, f25), name="resid_1_3")
        # self.resup1 = ResUp(s=2, filters=(f13, f23), name="resup_1")

        self.attention2 = DisparityAttentionv2(in_channels=f24, max_disparity=3)
        # self.resred1 = ResReduce(f23, name="resred_1")
        # self.resid4 = ResIdentity(filters=(f23, f23), name="resid_2_1")
        # self.resid5 = ResIdentity(filters=(f23, f23), name="resid_2_2")
        # self.resid6 = ResIdentity(filters=(f23, f23), name="resid_2_3")
        # self.resup2 = ResUp(s=2, filters=(f12 // 2, f22 // 2), name="resup_2")

        self.attention3 = DisparityAttentionv2(in_channels=f24, max_disparity=3)
        # self.resred2 = ResReduce(f22 // 2, name="resred_2")
        # self.resid7 = ResIdentity(filters=(f12 // 2, f22 // 2), name="resid_3_1")
        # self.resid8 = ResIdentity(filters=(f12 // 2, f22 // 2), name="resid_3_2")
        # self.resid9 = ResIdentity(filters=(f12 // 2, f22 // 2), name="resid_3_3")
        # self.resup3 = ResUp(s=2, filters=(f11 // 2, f21 // 2), name="resup_3")

        self.attention4 = DisparityAttentionv2(in_channels=f23, max_disparity=3)
        # self.resred3 = ResReduce(f21 // 2, name="resred_3")
        # self.resid10 = ResIdentity(filters=(f11 // 2, f21 // 2), name="resid_4_1")
        # self.resid11 = ResIdentity(filters=(f11 // 2, f21 // 2), name="resid_4_2")
        # self.resid12 = ResIdentity(filters=(f11 // 2, f21 // 2), name="resid_4_3")
        # self.resup4 = ResUp(s=2, filters=(f10 // 2, f20 // 2), name="resup_4")

        self.attention5 = DisparityAttentionv2(in_channels=f12+f12, max_disparity=3)
        self.resred = nn.Conv2d(
            f20+f12+f12,
            f20,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        # self.resred4 = ResReduce(f20 // 2, name="resred_4")

        self.head1 = nn.Conv2d(
            f20,
            f20,  # or 128?
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.bn = nn.BatchNorm2d(f20)

        self.disp_head = nn.ModuleList([
            nn.Conv2d(
                x,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ) for x in [f24, f24+f22, f24+f22, f23+f21, f20]
        ])

    def _make_identity_layer(self, dim):
        layer1 = ResidualBlock(dim, dim, stride=1)
        layer2 = ResidualBlock(dim, dim, stride=1)
        layer3 = ResidualBlock(dim, dim, stride=1)

        layers = (layer1, layer2, layer3)

        # self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, inputs):
        f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = inputs[0]
        f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = inputs[1]
        if DEBUG:
            print("f_l_1.shape", f_l_1.shape)
            print("f_r_1.shape", f_r_1.shape)

        before_head = f_l_5
        deep = True
        attention = []
        x_skip = f_l_1
        att_right1, x1, new_previous_weights = self.attention1([f_l_5, f_r_5])
        # attention.append(x)

        if DEBUG:
            print("x1.shape after attention1", x1.shape)


        # x = self.layer5(x1)

        # if DEBUG:
        #     print('x.shape after layer5 ', x.shape)
        # disp5 = self.disp_head[0](x)
        # x = self.resup1(x)
        # # x_skip = x

        # if DEBUG:
        #     print("x.shape after resup1", x.shape)

        att_right2, x2, pw = self.attention2([f_l_4, f_r_4], x1, pw)
        # attention.append(x)
        # x = torch.cat((x, x_skip), dim= 1)

        if DEBUG:
            print("x2.shape after attention2", x2.shape)

        # x = self.layer4(x2)
        # if DEBUG:
        #     print('x.shape after layer4 ', x.shape)
        # disp4 = self.disp_head[1](x)
        # x = self.resup2(x)
        # x_skip = x

        # if DEBUG:
        #     print("x.shape after resup2", x.shape)

        
        att_right3, x3, pw = self.attention3([f_l_3, f_r_3], x2, pw)
        # attention.append(x)
        # x = torch.cat((x, x_skip), dim=1)

        if DEBUG:
            print("x.shape after attention3", x3.shape)

        # x = self.layer3(x)
        # if DEBUG:
        #     print('x.shape after layer3 ', x.shape)
        # disp3 = self.disp_head[2](x)

        # if DEBUG:
        #     print("x.shape after layer3", x.shape)

        # x = self.resup3(x)
        # x_skip = x

        # if DEBUG:
        #     print("x.shape after resup3", x.shape)

        
        att_right4, x4, pw = self.attention4([f_l_2, f_r_2], x3, pw)
        # attention.append(x)
        # x = torch.cat((x, x_skip), dim=1)

        if DEBUG:
            print("x.shape after attention4", x4.shape)

        # x = self.layer2(x)
        # disp2 = self.disp_head[3](x)
        # x = self.resup4(x)

        # if DEBUG:
        #     print("x.shape after resup4", x.shape)

        # x_skip = x
        att_right5, x5, pw = self.attention5([f_l_1, f_r_1], x4, pw)
        # attention.append(x)
        x = torch.cat((x5, x_skip), dim=1)
        x = self.resred(x)
        x = self.head1(x)
        x = self.bn(x)
        stereo_disp = self.disp_head[4](x5)

        if DEBUG:
            print("x.shape after head1", x.shape)

        return stereo_disp, x4, x3, x2, x1



class DisparityAttentionv2(nn.Module):
    def __init__(self,in_channels, max_disparity=3):
        super(DisparityAttentionv2, self).__init__()
        self.max_disparity = max_disparity
        self.convs_block = self.create_conv_block(in_channels, in_channels, 3, 5)
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # Adjust based on input channels
            out_channels=max_disparity,
            kernel_size=(3, max_disparity + 2),
            padding="same"
        )


        self.softmax = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")


    def create_conv_block(self, in_channels, out_channels, kernel_size, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,  # Use in_channels for the first layer
                out_channels,
                kernel_size,
                padding='same' # Same padding
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    # def _make_up_layer():


    def forward(self, inputs, previous_weights=None, previous_disp=None):
        # Assume inputs is a list [left_image, right_image]
        left_image, right_image = inputs
        if DEBUG:
            print('left_image shape', left_image.shape)


        # if DEBUG:
        #     print('disp shape', disp.shape)
        # if previous_weights is not None:
        #     n = len(previous_weights)
        #     for i, pw in enumerate(previous_weights):
        #       if i==0:
        #         disp_down = torch.argmax(pw, dim=1, keepdim=True).type(torch.float32)
        #         disp_up = self.upsample(disp_down)*(2**(n-i))
        #         disp = disp_up
        #       else:
        #         disp_down = torch.argmax(pw, dim=1, keepdim=True).type(torch.float32)
        #         disp_up = self.upsample(disp_down)*(2**(n-i)) -1*(2**(n-i))
        #         disp = disp+disp_up
        # if DEBUG:
        #     print('disp before', disp)


        # Roll the right image to the left by max_disparity//2 pixels if previous_weights is given
        if previous_weights is not None:
            shift = -self.max_disparity // 2
        else:
            shift = -1
        right_image = torch.roll(right_image, shifts=shift, dims=3)

        # Add previous weights to the new list of weights and to the right image
        new_previous_weights = []
        if previous_weights is not None:
            if DEBUG:
                print(len(previous_weights))
            n = len(previous_weights) + 1
            for j, pw in enumerate(previous_weights):
                if DEBUG:
                    # print('pw',pw)
                    print('ri', right_image.shape)
                pw = self.upsample(pw)
                # print('pw after upsample', pw)
                if DEBUG:
                    print('pw after upsampling', pw.shape)
                new_previous_weights.append(pw)
                for i in range(self.max_disparity):
                    # Select the weights for disparity i and apply them to the right image shifted by i pixels
                    pw_right = pw[:,i:i+1,  :, :] * torch.roll(right_image, shifts=-i * (2 ** (n - j)), dims=3)
                    right_image += pw_right
                if DEBUG:
                    print('after applying previous weights', right_image.shape)


        # Stack left and right images
        stacked = torch.cat([left_image, right_image], dim=1)  # (B, C, H, W)
        if DEBUG:
            print('stacked', stacked.shape)

        # Compute disparity
        stacked = self.convs_block(stacked)
        disparity_map = self.conv(stacked)  # (B, max_disparity, H, W)
        if DEBUG:
            print('disparity_map', disparity_map.shape)

        # Apply softmax to get the weights
        weights = self.softmax(disparity_map)
        if DEBUG:
            print('weights', weights.shape)
        new_previous_weights.append(weights)

        # Compute attended right image
        attended_right = torch.zeros_like(right_image)
        for i in range(self.max_disparity):
            # Select the weights for disparity i and apply them to the right image shifted by i pixels
            # weighted_right = weights[:, :, :, i:i+1] * torch.roll(right_image, shifts=-i, dims=2)
            weighted_right = weights[:, i:i+1, :, :] * torch.roll(right_image, shifts=-i, dims=3)
            attended_right += weighted_right
        if DEBUG:
            print('attended_right', attended_right.shape)

        # disp_weights = torch.argmax(weights, dim=1)
        # disp = disp + disp_weights -1
        # if DEBUG:
        #   print('disp after', disp)
        #   print('disp shape', disp.shape)
        # Concatenate left image and attended right image
        weights2disp = torch.argmax(weights, dim=1, keepdim=True).type(torch.float32)
        print('w2d', weights2disp)
        if DEBUG:
            print('weights2disp', weights2disp.shape)


        if previous_disp is None:
          disp = weights2disp
        #   print('disp1', disp)
        else:
          previous_disp = self.upsample(previous_disp)*2
          disp = weights2disp + previous_disp-1
        #   print('disp2', disp)
        # x = torch.cat([left_image, attended_right], dim=1)  # (B, C, H, W)
        # x = [attended_right]
        # if DEBUG:
        #     print('torch cat (left_image, attended_right)', x.shape)

        return attended_right, disp, new_previous_weights
    