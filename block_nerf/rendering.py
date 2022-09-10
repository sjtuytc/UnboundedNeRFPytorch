import torch
from einops import rearrange, reduce, repeat
import pdb


def get_cone_mean_conv(t_samples, rays_o, rays_d, radii):
    # 得到每一个圆台区域内的均值和方差
    # t_samples:1024x65  rays_o:1024,3   radii:1024,1
    t0 = t_samples[..., :-1]  # 每个圆台区间的左侧
    t1 = t_samples[..., 1:]  # 每个圆台区间的右侧

    # 将圆台进行高斯近似
    # eq-7
    t_μ = (t0 + t1) / 2
    t_σ = (t1 - t0) / 2
    μ_t = t_μ + (2 * t_μ * t_σ ** 2) / (3 * t_μ ** 2 + t_σ ** 2)  # 真正的区间
    # 1024 x 64
    σ_t = (t_σ ** 2) / 3 - \
          (4 / 15) * \
          ((t_σ ** 4 * (12 * t_μ ** 2 - t_σ ** 2)) /
           (3 * t_μ ** 2 + t_σ ** 2) ** 2)  # σt
    σ_r = radii ** 2 * \
          (
                  (t_μ ** 2) / 4 + (5 / 12) * t_σ ** 2 - 4 /
                  15 * (t_σ ** 4) / (3 * t_μ ** 2 + t_σ ** 2)
          )
    # 求得eq8
    # mean = torch.unsqueeze(rays_d, dim=-2) * torch.unsqueeze(μ_t, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')
    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    mean = rays_o + rays_d * rearrange(μ_t, 'n1 n2 -> n1 n2 1')  # eq8
    # [1024,64,3]+[1024,1,3]*[1024,64,1]->[1024,64,3]
    # [B, 1, 3]*[B, N, 1] = [B, N, 3]

    rays_d = rays_d.squeeze()  # [1024,3]
    rays_o = rays_o.squeeze()  # [1024,3]
    # eq 16 mip-nerf
    dod = rays_d ** 2
    d2 = torch.sum(dod, dim=-1, keepdim=True) + 1e-10
    diagE = rearrange(σ_t, 'n1 c -> n1 c 1') * rearrange(dod, 'n1 c -> n1 1 c') + \
            rearrange(σ_r, 'n1 c -> n1 c 1') * \
            rearrange(1 - dod / d2, 'n1 c -> n1 1 c')

    return μ_t, mean, diagE  # [1024,64,3] [1024,64,3]


def sample_pdf(bins, weights, N_importance, alpha=1e-2):
    N_rays, N_samples_ = weights.shape

    # 要对weight进行blurpool
    weights_pad = torch.cat(
        [weights[..., :1], weights, weights[..., -1:]], dim=-1)
    # 第一个元素的max对应就是自己本身
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # prevent division by zero (don't do inplace op!)
    weights = weights + alpha
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1',
                           'sum')  # (N_rays, N_samples_)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    u = torch.linspace(0, 1, N_importance + 1, device=bins.device)
    u = u.expand(N_rays, N_importance + 1)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)  # inds_1中小于0的元素全部赋为0
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack(
        [below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled),
                      'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled),
                       'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < alpha] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / \
              denom * (bins_g[..., 1] - bins_g[..., 0])
    # 采样点
    return samples


def volume_rendering(rgbs=None, sigmas=None, z_vals=None, μ_t=None, type="train"):
    # 返回每个区间的权重
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # 把每一个圆台的中心看作是采样点?????
    # delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    # deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
    noise = torch.randn_like(sigmas)
    # (N_rays, N_samples_)
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 -
                   alphas + 1e-10], -1)  # （1,1-a1,1-a2）
    Ti = torch.cumprod(alphas_shifted[:, :-1], -1)
    weights = alphas * Ti  # (N_rays, N_samples_)
    # cumprod:返回元素的累计乘积 取[:-1]是因为eq3的上标是i-1
    # (N_rays), the accumulated opacity along the rays
    weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')
    # ∑Ti*(1-exp(-δσ))
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    results = {}
    results['transmittance'] = Ti  # 用于和Visibility网络做loss
    results['weights'] = weights
    results['opacity'] = weights_sum  # 乘以颜色或者z_vals就是像素颜色或者深度
    results['z_vals'] = z_vals

    if type == "test_coarse":  # 只需要weights！
        return results

    # weight:1024,64
    # rgb:1024,64,3
    # z_vals:1024,65
    rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')
                     * rgbs, 'n1 n2 c -> n1 c', 'sum')
    # depth_map = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')
    depth_map = reduce(weights * μ_t, 'n1 n2 -> n1', 'sum')
    # 深度图 深度图得用μ_t
    # rgb_map += 1 - weights_sum.unsqueeze(1) 白色背景才需要

    results[f'rgb'] = rgb_map
    results[f'depth'] = depth_map

    return results


def render_rays(models,
                embedding,  # embedding:{IPE: ;PE: ;appearance: }
                rays,
                ts,
                N_samples=64,
                N_importance=64,
                chunk=1024,
                type="train",
                use_disp=False):
    #   rays:[1024,10] [rays_o,rays_d,radii,exposure,near,far]
    N_rays = rays.shape[0]
    rays_o, rays_d, radii, exposure, near, far = torch.split(
        rays, [3, 3, 1, 1, 1, 1], dim=-1)
    # 先进行Coarse网络的处理
    z_steps = torch.linspace(
        0, 1, N_samples + 1, device=rays.device)  # 采样129个点，形成128个区域
    # 因为要采样64个区域，所以要采样65个点

    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
        # 2~6之间被分成了64个点：near+(far-near)*z_steps
    else:  # use linear sampling in disparity space
        # z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)
        z_vals = torch.exp(torch.log(near) * (1 - z_steps) + torch.log(far) * z_steps)

    # z_vals = near + (far - near) * z_steps
    z_vals_coarse = z_vals.expand(N_rays, N_samples + 1)  # 这句话有什么用

    #随机采样
    z_vals_mid = 0.5 * (z_vals_coarse[:, :-1] + z_vals_coarse[:, 1:])  # (N_rays, N_samples-1) interval mid points
    # get intervals between samples
    upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
    lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

    perturb_rand = 1 * torch.rand_like(z_vals)
    z_vals_coarse = lower + (upper - lower) * perturb_rand

    # 求粗采样阶段每个圆台区域的均值和方差
    μ_t_coarse, μ_coarse, diagE_coarse = get_cone_mean_conv(
        z_vals_coarse, rays_o, rays_d, radii)
    if type == "train":
        # 进行位置编码
        IPE = embedding['IPE']
        PE = embedding['PE']
        appearance_encoding = embedding['appearance']
        #########
        sample_coarse_encode = IPE(μ_coarse, diagE_coarse)  # 对位置进行编码
        sample_coarse_encode = rearrange(
            sample_coarse_encode, 'n1 n2 c -> (n1 n2) c')
        #########
        dir_coarse_encode = PE(rays_d)  # 对方向进行编码
        dir_coarse_encode = repeat(
            dir_coarse_encode, 'n1 c -> (n1 n2) c', n2=N_samples)
        #########
        exp_encode = PE(exposure)  # 对曝光率进行编码
        exp_coarse_encode = repeat(
            exp_encode, 'n1 c -> (n1 n2) c', n2=N_samples)
        # 外观嵌入向量
        appearance_encode = appearance_encoding(ts)  # [1024,32]
        appearance_coarse_encode = repeat(
            appearance_encode, 'n1 c -> (n1 n2) c', n2=N_samples)

        xyzdir_coarse_encode_f_σ = torch.cat(
            [sample_coarse_encode, dir_coarse_encode,
             exp_coarse_encode, appearance_coarse_encode], dim=-1)  # fσ的输入
        xyzdir_coarse_encode_f_v = torch.cat(
            [sample_coarse_encode, dir_coarse_encode], dim=-1)  # fv的输入

        out_coarse_rgb_sigma = []
        out_coarse_visibility = []
        for i in range(0, xyzdir_coarse_encode_f_σ.shape[0], chunk):
            result = models['block_model'](
                xyzdir_coarse_encode_f_σ[i:i + chunk])
            out_coarse_rgb_sigma.append(result)
        out_coarse_rgb_sigma = torch.cat(out_coarse_rgb_sigma, 0)
        out_coarse_rgb_sigma = rearrange(
            out_coarse_rgb_sigma, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples, c=4)

        # if type == "train":  # 只有train才考虑coarse阶段的visibility
        for i in range(0, xyzdir_coarse_encode_f_v.shape[0], chunk):
            result = models['visibility_model'](
                xyzdir_coarse_encode_f_v[i:i + chunk])
            out_coarse_visibility.append(result)
        out_coarse_visibility = torch.cat(out_coarse_visibility, 0)
        out_coarse_visibility = rearrange(out_coarse_visibility, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples,
                                          c=1)

        rgbs_coarse = out_coarse_rgb_sigma[..., :3]  # 每一点的像素RGB
        sigmas_coarse = out_coarse_rgb_sigma[..., 3]
        # 先对coarse网络进行渲染
        # 根据采样点进行渲染
        results_coarse = volume_rendering(
            rgbs_coarse, sigmas_coarse, z_vals_coarse, μ_t_coarse)

        ##################################
        # 处理fine网络
        # 首先逆采样
        z_vals_mid = 0.5 * (z_vals_coarse[:, :-1] + z_vals_coarse[:, 1:])
        z_vals_fine = sample_pdf(z_vals_mid, results_coarse['weights'][:, 1:-1].detach(),
                                 N_importance)  # 要采样129个点
        # 总共采样65+65个点 -> 130个点，形成129个圆台区域
        z_vals_fine = torch.sort(
            torch.cat([z_vals_coarse, z_vals_fine], -1), -1)[0]
        μ_t_fine, μ_fine, diagE_fine = get_cone_mean_conv(z_vals_fine, rays_o, rays_d, radii)

        sample_fine_encode = IPE(μ_fine, diagE_fine)  # 对位置进行编码
        sample_fine_encode = rearrange(
            sample_fine_encode, 'n1 n2 c -> (n1 n2) c')
        # 拓展到65+65-1=129个圆台区域上
        dir_fine_encode = PE(rays_d)
        dir_fine_encode = repeat(
            dir_fine_encode, 'n1 c -> (n1 n2) c', n2=N_samples + N_importance + 1)
        appearance_fine_encode = repeat(
            appearance_encode, 'n1 c -> (n1 n2) c', n2=N_samples + N_importance + 1)
        exp_fine_encode = repeat(
            exp_encode, 'n1 c -> (n1 n2) c', n2=N_samples + N_importance + 1)

        xyzdir_fine_encode_f_σ = torch.cat(
            [sample_fine_encode, dir_fine_encode,
             exp_fine_encode, appearance_fine_encode], dim=-1)
        xyzdir_fine_encode_f_v = torch.cat(
            [sample_fine_encode, dir_fine_encode], dim=-1)

        out_fine_rgb_sigma = []
        out_fine_visibility = []
        for i in range(0, xyzdir_fine_encode_f_σ.shape[0], chunk):
            result = models['block_model'](xyzdir_fine_encode_f_σ[i:i + chunk])
            out_fine_rgb_sigma.append(result)
        out_fine_rgb_sigma = torch.cat(out_fine_rgb_sigma, 0)
        out_fine_rgb_sigma = rearrange(out_fine_rgb_sigma, '(n1 n2) c -> n1 n2 c', n1=N_rays,
                                       n2=N_samples + N_importance + 1, c=4)

        for i in range(0, xyzdir_fine_encode_f_v.shape[0], chunk):
            result = models['visibility_model'](
                xyzdir_fine_encode_f_v[i:i + chunk])
            out_fine_visibility.append(result)
        out_fine_visibility = torch.cat(out_fine_visibility, 0)
        out_fine_visibility = rearrange(out_fine_visibility, '(n1 n2) c -> n1 n2 c', n1=N_rays,
                                        n2=N_samples + N_importance + 1, c=1)

        rgbs_fine = out_fine_rgb_sigma[..., :3]
        sigmas_fine = out_fine_rgb_sigma[..., 3]
        # 先对coarse网络进行渲染
        # 根据采样点进行渲染
        results_fine = volume_rendering(rgbs_fine, sigmas_fine, z_vals_fine, μ_t_fine)

        result = {}
        result['rgb_coarse'] = results_coarse['rgb']
        result['rgb_fine'] = results_fine['rgb']
        result['depth_fine'] = results_fine['depth']
        result['transmittance_coarse_real'] = results_coarse['transmittance']
        result['transmittance_fine_real'] = results_fine['transmittance']
        result['transmittance_coarse_vis'] = out_coarse_visibility.squeeze()
        result['transmittance_fine_vis'] = out_fine_visibility.squeeze()

        # rearrange(results_fine['transmittance'],"n1 n2 -> n1 n2 1").shape
        return result

    else:  # 如果是test和validation的时候
        # 进行位置编码
        IPE = embedding['IPE']
        PE = embedding['PE']
        appearance_encoding = embedding['appearance']
        exp_encode = PE(exposure)  # 对曝光率进行编码
        appearance_encode = appearance_encoding(ts)  # [1024,32]
        #########
        sample_coarse_encode = IPE(μ_coarse, diagE_coarse)  # 对位置进行编码
        sample_coarse_encode = rearrange(
            sample_coarse_encode, 'n1 n2 c -> (n1 n2) c')

        xyzdir_coarse_encode_f_σ = sample_coarse_encode

        out_coarse_sigma = []
        for i in range(0, xyzdir_coarse_encode_f_σ.shape[0], chunk):
            result = models['block_model'](
                xyzdir_coarse_encode_f_σ[i:i + chunk], sigma_only=True)
            out_coarse_sigma.append(result)
        out_coarse_sigma = torch.cat(out_coarse_sigma, 0)
        out_coarse_sigma = rearrange(
            out_coarse_sigma, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples, c=1)

        sigmas_coarse = out_coarse_sigma.squeeze()
        # 先对coarse网络进行渲染
        # 根据采样点进行渲染
        results_coarse = volume_rendering(
            sigmas=sigmas_coarse, z_vals=z_vals_coarse, type="test_coarse")  # 只需要result的weight进行采样

        ##################################
        # 处理fine网络
        # 首先逆采样
        #########
        z_vals_mid = 0.5 * (z_vals_coarse[:, :-1] + z_vals_coarse[:, 1:])
        z_vals_fine = sample_pdf(z_vals_mid, results_coarse['weights'][:, 1:-1].detach(),
                                 N_importance)  # 要采样129个点
        # 总共采样65+65个点 -> 130个点，形成129个圆台区域
        z_vals_fine = torch.sort(torch.cat([z_vals_coarse, z_vals_fine], -1), -1)[0]
        μ_t_fine, μ_fine, diagE_fine = get_cone_mean_conv(z_vals_fine, rays_o, rays_d, radii)

        sample_fine_encode = IPE(μ_fine, diagE_fine)  # 对位置进行编码
        sample_fine_encode = rearrange(
            sample_fine_encode, 'n1 n2 c -> (n1 n2) c')
        # 拓展到65+65-1=129个圆台区域上
        dir_fine_encode = PE(rays_d)
        dir_fine_encode = repeat(
            dir_fine_encode, 'n1 c -> (n1 n2) c', n2=N_samples + N_importance + 1)
        appearance_fine_encode = repeat(
            appearance_encode, 'n1 c -> (n1 n2) c', n2=N_samples + N_importance + 1)
        exp_fine_encode = repeat(
            exp_encode, 'n1 c -> (n1 n2) c', n2=N_samples + N_importance + 1)

        xyzdir_fine_encode_f_σ = torch.cat(
            [sample_fine_encode, dir_fine_encode,
             exp_fine_encode, appearance_fine_encode], dim=-1)
        xyzdir_fine_encode_f_v = torch.cat(
            [sample_fine_encode, dir_fine_encode], dim=-1)

        out_fine_rgb_sigma = []
        out_fine_visibility = []
        for i in range(0, xyzdir_fine_encode_f_σ.shape[0], chunk):
            result = models['block_model'](xyzdir_fine_encode_f_σ[i:i + chunk])
            out_fine_rgb_sigma.append(result)
        out_fine_rgb_sigma = torch.cat(out_fine_rgb_sigma, 0)
        out_fine_rgb_sigma = rearrange(out_fine_rgb_sigma, '(n1 n2) c -> n1 n2 c', n1=N_rays,
                                       n2=N_samples + N_importance + 1, c=4)

        for i in range(0, xyzdir_fine_encode_f_v.shape[0], chunk):
            result = models['visibility_model'](
                xyzdir_fine_encode_f_v[i:i + chunk])
            out_fine_visibility.append(result)
        out_fine_visibility = torch.cat(out_fine_visibility, 0)
        out_fine_visibility = rearrange(out_fine_visibility, '(n1 n2) c -> n1 n2 c', n1=N_rays,
                                        n2=N_samples + N_importance + 1, c=1)

        rgbs_fine = out_fine_rgb_sigma[..., :3]
        sigmas_fine = out_fine_rgb_sigma[..., 3]
        # 先对coarse网络进行渲染
        # 根据采样点进行渲染
        results_fine = volume_rendering(
            rgbs_fine, sigmas_fine, z_vals_fine, μ_t_fine)

        result = {}
        result['rgb_fine'] = results_fine['rgb']
        result['depth_fine'] = results_fine['depth']
        result['transmittance_fine_vis'] = out_fine_visibility.squeeze()

        # rearrange(results_fine['transmittance'],"n1 n2 -> n1 n2 1").shape
        return result


def test_train():
    rays = torch.rand(1024, 11)
    rays_test = rays[:, :10]
    ts_test = torch.ones_like(rays[:, 10]).long()

    block_model = Block_NeRF()
    visibility_model = Visibility()
    model = {
        "block_model": block_model,
        "visibility_model": visibility_model
    }
    embedding = {}
    IPE = InterPosEmbedding(N_freqs=10)
    embedding['IPE'] = IPE
    PE = PosEmbedding(N_freqs=4)
    embedding['PE'] = PE
    embedding_appearance = torch.nn.Embedding(1500, 32)
    embedding['appearance'] = embedding_appearance

    result = render_rays(models=model,
                         embedding=embedding,
                         rays=rays_test,
                         ts=ts_test)
    print()


def test_test():
    rays = torch.rand(1024, 11)
    rays_test = rays[:, :10]
    ts_test = torch.ones_like(rays[:, 10]).long()

    block_model = Block_NeRF()
    visibility_model = Visibility()
    model = {
        "block_model": block_model,
        "visibility_model": visibility_model
    }
    embedding = {}
    IPE = InterPosEmbedding(N_freqs=10)
    embedding['IPE'] = IPE
    PE = PosEmbedding(N_freqs=4)
    embedding['PE'] = PE
    embedding_appearance = torch.nn.Embedding(1500, 32)
    embedding['appearance'] = embedding_appearance

    result = render_rays(models=model, embedding=embedding,
                         rays=rays_test, ts=ts_test, type="val")
    print(result)


if __name__ == "__main__":
    from Block_NeRF import *
    test_train()
