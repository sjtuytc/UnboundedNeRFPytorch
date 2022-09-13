import torch
from torch import nn


class BlockNeRFLoss(nn.Module):
    def __init__(self, λ_u=0.01, Visi_loss=1e-2):
        super(BlockNeRFLoss, self).__init__()
        self.λ_u = λ_u
        self.Visi_loss = Visi_loss

    def forward(self, inputs, targets):
        loss = {}
        # RGB
        loss['rgb_coarse'] = self.λ_u * ((inputs['rgb_coarse'] - targets[..., :3]) ** 2).mean()
        loss['rgb_fine'] = ((inputs['rgb_fine'] - targets[..., :3]) ** 2).mean()
        # visibility
        loss["transmittance_coarse"] = self.λ_u * self.Visi_loss * ((inputs['transmittance_coarse_real'].detach() -
                                                                     inputs['transmittance_coarse_vis'].squeeze()) ** 2).mean()
        loss["transmittance_fine"] = self.Visi_loss * ((inputs['transmittance_fine_real'].detach() - inputs[
            'transmittance_fine_vis'].squeeze()) ** 2).mean()

        return loss


class InterPosEmbedding(nn.Module):
    def __init__(self, N_freqs=10):
        super(InterPosEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        # [2^0,2^1,...,2^(n-1)]: for sin
        self.freq_band_1 = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        # [4^0,4^1,...,4^(n-1)]: for diag(∑)
        self.freq_band_2 = self.freq_band_1 ** 2

    def forward(self, μ, diagE):
        # exmbeds [μ,diagE] -> [sin(μ)*exp(-1/2)*diag(∑γ),cos(μ)*exp(-1/2)*diag(∑γ),...,
        # sin(2^(L-1)*μ)*exp(-1/2)*4^(L-1)*diag(∑)]
        sin_out = []
        sin_cos = []
        for freq in self.freq_band_1:
            for func in self.funcs:
                sin_cos.append(func(freq * μ))
            sin_out.append(sin_cos)
            sin_cos = []
        # sin_out:list:[sin(μ),cos(μ)]
        diag_out = []
        for freq in self.freq_band_2:
            diag_out.append(freq * diagE)
        # diag_out:list:[4^(L-1)*diag(∑)]
        out = []
        for sc_γ, diag_Eγ in zip(sin_out, diag_out):
            # torch.exp(-0.5 * x_var) * torch.sin(x)
            for sin_cos in sc_γ:  # [sin,cos]
                out.append(sin_cos * torch.exp(-0.5 * diag_Eγ))
        return torch.cat(out, -1)


class PosEmbedding(nn.Module):
    def __init__(self, N_freqs):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        # [2^0,2^1,...,2^(n-1)]
        self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)

    def forward(self, x):
        out = []
        for freq in self.freq_bands:  # [2^0,2^1,...,2^(n-1)]
            for func in self.funcs:
                out += [func(freq * x)]
        # !!!!相当于63维，多了三个基础坐标——>[x,y,z,sin(2^0Πpi),cos.......]
        # xyz——>63,dir——>27
        return torch.cat(out, -1)  # 变成一个63的元素

class Block_NeRF(nn.Module):
    def __init__(self, D=8, W=256, skips=[4],
                 in_channel_xyz=60, in_channel_dir=24,
                 in_channel_exposure=8,  # exposure is in 1d and dirs are in 3d
                 in_channel_appearance=32,
                 add_apperance=True,
                 add_exposure=False):
        # input：[xyz60,dir24,exposure24,appearance24]
        super(Block_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channel_xyz = in_channel_xyz
        self.in_channel_dir = in_channel_dir
        self.in_channel_exposure = in_channel_exposure
        self.in_channel_appearance = in_channel_appearance
        self.add_appearance = add_apperance
        self.add_exposure = add_exposure

        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channel_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channel_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'xyz_encoding_{i + 1}', layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        input_channel = W + in_channel_dir
        if add_apperance:
            input_channel += in_channel_appearance
        if add_exposure:
            input_channel += in_channel_exposure
        # 3层128
        self.dir_encoding = nn.Sequential(  # RGB由dir,Exposure,Appearance决定
            nn.Linear(
                input_channel,
                W // 2
            ), nn.ReLU(True),
            nn.Linear(W // 2, W // 2), nn.ReLU(True),
            nn.Linear(W // 2, W // 2), nn.ReLU(True)
        )

        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        if sigma_only:
            input_xyz = x
        else:
            input_xyz, input_dir, input_exp, input_appear = torch.split(x, [self.in_channel_xyz, self.in_channel_dir,
                                                                        self.in_channel_exposure,
                                                                        self.in_channel_appearance], dim=-1)
        xyz = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz = torch.cat([xyz, input_xyz], dim=-1)
            xyz = getattr(self, f'xyz_encoding_{i + 1}')(xyz)

        static_sigma = self.static_sigma(xyz)
        if sigma_only:
            return static_sigma

        xyz_feature = self.xyz_encoding_final(xyz)
        input_xyz_feature = torch.cat([xyz_feature, input_dir], dim=-1)
        if self.add_exposure:
            input_xyz_feature = torch.cat([input_xyz_feature, input_exp], dim=-1)
        if self.add_appearance:
            input_xyz_feature = torch.cat([input_xyz_feature, input_appear], dim=-1)
        
        dir_encoding = self.dir_encoding(input_xyz_feature)

        static_rgb = self.static_rgb(dir_encoding)
        static_rgb_sigma = torch.cat([static_rgb, static_sigma], dim=-1)

        return static_rgb_sigma


class Visibility(nn.Module):
    def __init__(self,
                 in_channel_xyz=60, in_channel_dir=24,
                 W=128):
        super(Visibility, self).__init__()
        self.in_channel_xyz = in_channel_xyz
        self.in_channel_dir = in_channel_dir

        self.vis_encoding = nn.Sequential(
            nn.Linear(in_channel_xyz + in_channel_dir, W), nn.ReLU(True),
            nn.Linear(W, W), nn.ReLU(True),
            nn.Linear(W, W), nn.ReLU(True),
            nn.Linear(W, W), nn.ReLU(True),
        )
        self.visibility = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

    def forward(self, x):
        vis_encode = self.vis_encoding(x)
        visibility = self.visibility(vis_encode)
        return visibility


def test_Block():
    # xyz,dir,exposure,appearance
    test_data = torch.rand([1024, 64, 60 + 24 + 8 + 32])
    model = Block_NeRF()
    result = model(test_data)
    print(result.shape)


def test_Vis():
    test_data = torch.rand([1024, 64, 84])
    model = Visibility()
    result = model(test_data)
    print()


if __name__ == '__main__':
    # test_Vis()
    test_Block()