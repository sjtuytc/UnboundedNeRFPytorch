import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import pdb
import time
import torch
# import random
from jax import random
import torch.nn as nn
import numpy as np


class VG(nn.Module):
    # the VG operator
    def __init__(self, grid_len=1000):
        super(VG, self).__init__()
        self.grid_len = grid_len
        self.interval_num = grid_len - 1
        axis_coord = np.array([0 + i * 1 / grid_len for i in range(grid_len)])
        self.ms_x, self.ms_t = np.meshgrid(axis_coord, axis_coord)  # x and t of the grid
        x_coord = np.ravel(self.ms_x).reshape(-1,1)
        t_coord = np.ravel(self.ms_t).reshape(-1,1)
        self.x_coord = torch.tensor(x_coord).float()
        self.t_coord = torch.tensor(t_coord).float()
        axis_index = np.array([i for i in range(grid_len)])
        ms_x, ms_t = np.meshgrid(axis_index, axis_index)
        x_ind = np.ravel(ms_x).reshape(-1,1)
        t_ind = np.ravel(ms_t).reshape(-1,1)
        self.x_ind = torch.tensor(x_ind).long()
        self.t_ind = torch.tensor(t_ind).long()
        self.voxel = nn.Parameter(torch.rand((grid_len)), requires_grad=True)
    
    def forward(self,):  # calculate GTK
        # the data is [0, 1/data_point_num, 2/data_point_num, ..., 1]
        jacobian_y_w = np.zeros((data_point_num, self.grid_len))
        for idx in range(data_point_num):
            real_x = idx / data_point_num
            left_grid = int(real_x // (1 / self.grid_len))
            right_grid = left_grid + 1
            if left_grid > 0:
                jacobian_y_w[idx][left_grid] = abs(real_x - right_grid * 1 / self.grid_len) * self.grid_len
            if right_grid < self.grid_len:
                jacobian_y_w[idx][right_grid] = abs(real_x - left_grid * 1 / self.grid_len) * self.grid_len
        jacobian_y_w_transpose = np.transpose(jacobian_y_w)
        result_matrix = np.matmul(jacobian_y_w, jacobian_y_w_transpose)
        return result_matrix

    def one_d_regress(self, x_train, x_test, y_train, y_test_gt):
        train_loss = 0
        for idx, one_x in enumerate(x_train):
            left_grid = int(one_x // (1 / self.interval_num))
            right_grid = left_grid + 1
            left_value = self.voxel[left_grid]
            right_value = self.voxel[right_grid]
            left_weight = abs(one_x - right_grid * 1 / self.interval_num) * self.interval_num
            right_weight = abs(one_x - left_grid * 1 / self.interval_num) * self.interval_num
            y_pred = left_value * left_weight + right_value * right_weight
            y_pred = torch.sigmoid(y_pred)
            train_loss += torch.nn.functional.mse_loss(y_pred, torch.tensor(y_train[idx]).float())
        
        y_test = []
        test_loss = []
        for idx, one_x in enumerate(x_test):
            left_grid = int(one_x // (1 / self.interval_num))
            right_grid = left_grid + 1
            left_value = self.voxel[left_grid]
            right_value = self.voxel[right_grid]
            left_weight = abs(one_x - right_grid * 1 / self.interval_num) * self.interval_num
            right_weight = abs(one_x - left_grid * 1 / self.interval_num) * self.interval_num
            y_pred = left_value * left_weight + right_value * right_weight
            y_pred = torch.sigmoid(y_pred)
            y_test.append(y_pred.item())
            test_loss.append((y_pred.item() - y_test_gt[idx]) ** 2)
        test_loss = np.mean(test_loss)
        return train_loss, test_loss, y_test


class FG(nn.Module):
    # the FG operator
    def __init__(self, grid_len=1000, band_num=10):
        super(FG, self).__init__()
        self.grid_len = grid_len
        self.interval_num = self.grid_len - 1
        self.band_num = band_num
        axis_coord = np.array([0 + i * 1 / grid_len for i in range(grid_len)])
        self.ms_x, self.ms_t = np.meshgrid(axis_coord, axis_coord)  # x and t of the grid
        x_coord = np.ravel(self.ms_x).reshape(-1,1)
        t_coord = np.ravel(self.ms_t).reshape(-1,1)
        self.x_coord = torch.tensor(x_coord).float()
        self.t_coord = torch.tensor(t_coord).float()
        axis_index = np.array([i for i in range(grid_len)])
        ms_x, ms_t = np.meshgrid(axis_index, axis_index)
        x_ind = np.ravel(ms_x).reshape(-1,1)
        t_ind = np.ravel(ms_t).reshape(-1,1)
        self.x_ind = torch.tensor(x_ind).long()
        self.t_ind = torch.tensor(t_ind).long()
        self.voxel = nn.Parameter(torch.rand((grid_len * (self.band_num + 1))), requires_grad=True)
    
    def gamma_x_i(self, x, i):
        if i%2 == 0:
            raw_fourier = np.sin((2^(i // 2)) * np.pi * x)
        else:
            raw_fourier = np.cos((2^(i // 2)) * np.pi * x)
        fourier = (raw_fourier + 1) / 2   # to [0, 1]
        return fourier
        
    def forward(self,):  # calculate GTK
        jacobian_y_w = np.zeros((data_point_num, self.grid_len * self.band_num))
        for idx in range(data_point_num):  # for all data points
            real_x = idx / data_point_num   # the real x value
            for jdx in range(self.band_num):
                fourier = self.gamma_x_i(real_x, jdx)
                left_grid = int(fourier // (1 / self.grid_len))
                right_grid = right_grid = left_grid + 1
                if left_grid > 0:
                    jacobian_y_w[idx][self.grid_len * jdx + left_grid] = abs(fourier - right_grid * 1 / self.grid_len) * self.grid_len
                if right_grid < self.grid_len:
                    jacobian_y_w[idx][self.grid_len * jdx + right_grid] = abs(fourier - left_grid * 1 / self.grid_len) * self.grid_len
        jacobian_y_w_transpose = np.transpose(jacobian_y_w)
        result_matrix = np.matmul(jacobian_y_w, jacobian_y_w_transpose)
        return result_matrix
    
    def one_d_regress(self, x_train, x_test, y_train, y_test_gt):
        train_loss = 0
        for idx, one_x in enumerate(x_train):
            y_pred = 0
            for jdx in range(self.band_num):
                fourier = self.gamma_x_i(one_x, jdx)
                left_grid = int(fourier * self.interval_num)
                right_grid = left_grid + 1
                left_value = self.voxel[self.grid_len * jdx + left_grid]
                right_value = self.voxel[self.grid_len * jdx + right_grid]
                left_weight = abs(fourier - right_grid * 1 / self.interval_num) * self.interval_num
                right_weight = abs(fourier - left_grid * 1 / self.interval_num) * self.interval_num
                assert abs(left_weight + right_weight - 1) < 0.0001
                y_pred += left_value * left_weight + right_value * right_weight
            y_pred /= self.band_num
            y_pred = torch.sigmoid(y_pred)
            train_loss += torch.nn.functional.mse_loss(y_pred, torch.tensor(y_train[idx]).float())
        
        y_test = []
        test_loss = []
        for idx, one_x in enumerate(x_test):
            y_pred = 0
            for jdx in range(self.band_num):
                fourier = self.gamma_x_i(one_x, jdx)
                left_grid = int(fourier * self.interval_num)
                right_grid = left_grid + 1
                left_value = self.voxel[self.grid_len * jdx + left_grid]
                right_value = self.voxel[self.grid_len * jdx + right_grid]
                left_weight = abs(fourier - right_grid * 1 / self.interval_num) * self.interval_num
                right_weight = abs(fourier - left_grid * 1 / self.interval_num) * self.interval_num
                y_pred += left_value * left_weight + right_value * right_weight
            y_pred /= self.band_num
            y_pred = torch.sigmoid(y_pred)
            y_test.append(y_pred.item())
            test_loss.append((y_pred.item() - y_test_gt[idx]) ** 2)
        test_loss = np.mean(test_loss)
        return train_loss, test_loss, y_test

# hyperparameters
title_offset = -0.4
data_point_num = 100
grid_len = 10
freq_num = 10
colors_k = np.array([[0.8872, 0.4281, 0.1875],
    [0.8136, 0.6844, 0.0696],
    [0.2634, 0.6634, 0.4134],
    [0.0943, 0.5937, 0.8793],
    [0.3936, 0.2946, 0.6330],
    [0.7123, 0.2705, 0.3795]])
linewidth = 3
line_alpha = .8

# begin plot 
fig3 = plt.figure(constrained_layout=True, figsize=(4, 2))
gs = fig3.add_gridspec(1, 2, width_ratios=[1, 1])
# 100 * 100 datapoints, 10*10 params (grid_len=10)
test_vg = VG(grid_len=grid_len * freq_num)
vg_gtk = test_vg()
vg_gtk = (vg_gtk - vg_gtk.min()) / vg_gtk.max()
ax = fig3.add_subplot(gs[0, 0])
ax.imshow(vg_gtk)
ax.set_xticks([*range(0, 100, 20)] + [100])
ax.set_yticks([*range(0, 100, 20)] + [100])
ax.set_title('(a) VG GTK', y=title_offset)

ax = fig3.add_subplot(gs[0, 1])
test_fg = FG(grid_len=grid_len, band_num=freq_num)
fg_gtk = test_fg()
fg_gtk = (fg_gtk - fg_gtk.min()) / fg_gtk.max()
ax.imshow(fg_gtk)
ax.set_xticks([*range(0, 100, 20)] + [100])
ax.set_yticks([*range(0, 100, 20)] + [100])
ax.set_title('(b) FG GTK', y=title_offset)

# generate figures
plt.savefig("figures/vg_fg_gtk.jpg", dpi=800)
plt.savefig("figures/vg_fg_gtk.pdf", format="pdf")
pdb.set_trace()


def sample_random_signal(key, decay_vec):
  N = decay_vec.shape[0]
#   raw = random.normal(key, [N, 2]) @ np.array([1, 1j])
  raw = np.random.normal(size=[N, 2]) @ np.array([1, 1j])
  signal_f = raw * decay_vec
  signal = np.real(np.fft.ifft(signal_f))
  return signal


def sample_random_powerlaw(key, N, power):
  coords = np.float32(np.fft.ifftshift(1 + N//2 - np.abs(np.fft.fftshift(np.arange(N)) - N//2)))
  decay_vec = coords ** (-power)
  decay_vec = np.array(decay_vec)
  decay_vec[N//4:] = 0
  return sample_random_signal(key, decay_vec)


##  Fitting experiments
# hyperparameters
rand_key = np.array([0, 0], dtype=np.uint32)
train_num = 7
sample_interval = 4
data_power = 0.5
lr = 1


# setup data
x_test = np.float32(np.linspace(0, 1., train_num * sample_interval, endpoint=False))
x_train = x_test[::sample_interval]
# s = sample_random_powerlaw(rand_key, train_num * sample_interval, data_power) 
signal = np.array([np.sin(x / (train_num*sample_interval) * 2 * np.pi) for x in range(train_num*sample_interval)])
signal = (signal-signal.min()) / (signal.max()-signal.min())

y_train = signal[::sample_interval]
y_test_gt = signal


# build models and train them
def train_model(one_model):
    # training and testing VG
    optimizer = torch.optim.Adam(one_model.parameters(), lr=lr)
    iterations = 150
    epoch_iter = tqdm(range(iterations))
    for epoch in epoch_iter:
        optimizer.zero_grad() # to make the gradients zero
        train_loss, test_loss, test_y = one_model.one_d_regress(x_train, x_test, y_train, y_test_gt)
        train_loss.backward() # This is for computing gradients using backward propagation
        optimizer.step()      # This is equivalent to: theta_new = theta_old - alpha * derivative of J w.r.t theta
        epoch_iter.set_description(f"Training loss: {train_loss.item()}; Testing Loss: {test_loss}")
    return train_loss, test_loss, test_y

freq_num = 3
test_vg_small = VG(grid_len=10 * freq_num)
test_vg_large = VG(grid_len=100 * freq_num)
test_fg_small = FG(grid_len=10, band_num=freq_num)
test_fg_large = FG(grid_len=100, band_num=freq_num)
train_loss, test_loss, test_y_vg_small = train_model(test_vg_small)
train_loss, test_loss, test_y_fg_small = train_model(test_fg_small)


ax = fig3.add_subplot(gs[0, 2])
ax.plot(x_test, signal, label='Target signal', color='k', linewidth=1, alpha=line_alpha, zorder=1)
ax.plot(x_test, test_y_vg_small, label='Learned by VG', color=colors_k[1], linewidth=1, alpha=line_alpha, zorder=1)
ax.plot(x_test, test_y_fg_small, label='Learned by FG', color=colors_k[2], linewidth=1, alpha=line_alpha, zorder=1)
ax.scatter(x_train, y_train, color='w', edgecolors='k', linewidths=1, s=20, linewidth=1, label='Training points', zorder=2)
ax.set_title('(c) 1D Regression', y=title_offset)
ax.set_xticks(np.linspace(0.0, 1.0, num=5, endpoint=True))

# ax.set_xticks([])
# ax.set_yticks([])
# ax.legend(loc='upper right', ncol=2)
ax.legend(loc='upper left', bbox_to_anchor=(1.03, 0.78), handlelength=1)

# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True, which='both', alpha=.3)


# generate figures
plt.savefig("figures/final_vg_fg.jpg", dpi=800)
plt.savefig("figures/final_vg_fg.pdf", format="pdf")
pdb.set_trace()


# # unused codes

# fplot = lambda x : np.fft.fftshift(np.log10(np.abs(np.fft.fft(x))))
# ax = fig3.add_subplot(gs[0, 2])
# vg_spec = 10**fplot(vg_gtk)
# fg_spec = 10**fplot(fg_gtk)
# w_vg, v_vg = np.linalg.eig(vg_gtk)
# w_fg, v_fg = np.linalg.eig(fg_gtk)
# plt.semilogy(vg_spec[0], label="VG", color=colors_k[0], alpha=line_alpha, linewidth=linewidth)
# plt.semilogy(fg_spec[0], label="FG", color=colors_k[1], alpha=line_alpha, linewidth=linewidth)
# # ax.plot(np.linspace(-.5, .5, 10100, endpoint=True), np.append(vg_spec, vg_spec[0]), label="vg", color=colors_k[0], alpha=line_alpha, linewidth=linewidth)
# ax.set_title('(c) GTK Fourier spectrum', y=title_offset)

# pdb.set_trace()



# import jax
# from jax import random, grad, jit, vmap
# from jax.config import config
# from jax.lib import xla_bridge
# import jax.numpy as np
# import neural_tangents as nt
# from neural_tangents import stax
# from jax.example_libraries import optimizers
# import os


# # Utils

# fplot = lambda x : np.fft.fftshift(np.log10(np.abs(np.fft.fft(x))))

# # Signal makers

# def sample_random_signal(key, decay_vec):
#   N = decay_vec.shape[0]
#   raw = np.random.normal(key, [N, 2]) @ np.array([1, 1j])
#   signal_f = raw * decay_vec
#   signal = np.real(np.fft.ifft(signal_f))
#   return signal

# def sample_random_powerlaw(key, N, power):
#   coords = np.float32(np.fft.ifftshift(1 + N//2 - np.abs(np.fft.fftshift(np.arange(N)) - N//2)))
#   decay_vec = coords ** -power
#   decay_vec = np.array(decay_vec)
#   decay_vec[N//4:] = 0
#   return sample_random_signal(key, decay_vec)


# # Network 

# def make_network(num_layers, num_channels, ntk_params=True, num_outputs=1):
#   layers = []
#   for i in range(num_layers-1):
#     if ntk_params:
#         layers.append(stax.Dense(num_channels, parameterization='standard'))
#     else:
#         layers.append(stax.Dense(num_channels, parameterization='standard'))
#     layers.append(stax.Relu(do_backprop=True))
#   layers.append(stax.Dense(num_outputs, parameterization='standard'))
#   return stax.serial(*layers)

# # Encoding 

# def compute_ntk(x, avals, bvals, kernel_fn):
#     x1_enc = input_encoder(x, avals, bvals)
#     x2_enc = input_encoder(np.array([0.], dtype=np.float32), avals, bvals)
#     out = np.squeeze(kernel_fn(x1_enc, x2_enc, 'ntk'))
#     return out


# input_encoder = lambda x, a, b: np.concatenate([a * np.sin((2.*np.pi*x[...,None]) * b), 
#                                                 a * np.cos((2.*np.pi*x[...,None]) * b)], axis=-1) / np.linalg.norm(a)


# def predict_psnr_basic(kernel_fn, train_fx, test_fx, train_x, train_y, test_x, test_y, t_final, eta=None):  
#   g_dd = kernel_fn(train_x, train_x, 'ntk')
#   g_td = kernel_fn(test_x, train_x, 'ntk')
#   train_predict_fn = nt.predict.gradient_descent_mse(g_dd, train_y[...,None], g_td)
#   train_theory_y, test_theory_y = train_predict_fn(t_final, train_fx[...,None], test_fx[...,None])

#   calc_psnr = lambda f, g: -10. * np.log10(np.mean((f-g)**2))
#   return calc_psnr(test_y, test_theory_y[:,0]), calc_psnr(train_y, train_theory_y[:,0])

# predict_psnr_basic = jit(predict_psnr_basic, static_argnums=(0,))


# def train_model(rand_key, network_size, lr, iters, 
#                 train_input, test_input, test_mask, optimizer, ab, name=''):
#     if ab is None:
#         ntk_params = False
#     else:
#         ntk_params = True
#     init_fn, apply_fn, kernel_fn = make_network(*network_size, ntk_params=ntk_params)

#     if ab is None:
#         run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, x[...,None] - .5)))
#     else:
#         run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, input_encoder(x, *ab))))
#     model_loss = jit(lambda params, ab, x, y: .5 * np.sum((run_model(params, ab, x) - y) ** 2))
#     model_psnr = jit(lambda params, ab, x, y: -10 * np.log10(np.mean((run_model(params, ab, x) - y) ** 2)))
#     model_grad_loss = jit(lambda params, ab, x, y: jax.grad(model_loss)(params, ab, x, y))

#     opt_init, opt_update, get_params = optimizer(lr)
#     opt_update = jit(opt_update)

#     if ab is None:
#         _, params = init_fn(rand_key, (-1, 1))
#     else:
#         _, params = init_fn(rand_key, (-1, input_encoder(train_input[0], *ab).shape[-1]))
#     opt_state = opt_init(params)

#     pred0 = run_model(get_params(opt_state), ab, test_input[0])
#     pred0_f = np.fft.fft(pred0)

#     train_psnrs = []
#     test_psnrs = []
#     theories = []
#     xs = []
#     errs = []
#     for i in tqdm(range(iters), desc=name):
#         opt_state = opt_update(i, model_grad_loss(get_params(opt_state), ab, *train_input), opt_state)

#         if i % 20 == 0:
#             train_psnr = model_psnr(get_params(opt_state), ab, *train_input)
#             test_psnr = model_psnr(get_params(opt_state), ab, test_input[0][test_mask], test_input[1][test_mask])
#             if ab is None:
#                 train_fx = run_model(get_params(opt_state), ab, train_input[0])
#                 test_fx = run_model(get_params(opt_state), ab, test_input[0][test_mask])
#                 theory = predict_psnr_basic(kernel_fn, train_fx, test_fx, train_input[0][...,None]-.5, train_input[1], test_input[0][test_mask][...,None], test_input[1][test_mask], i*lr)
#             else:
#                 test_x = input_encoder(test_input[0][test_mask], *ab)
#                 train_x = input_encoder(train_input[0], *ab)

#                 train_fx = run_model(get_params(opt_state), ab, train_input[0])
#                 test_fx = run_model(get_params(opt_state), ab, test_input[0][test_mask])
#                 theory = predict_psnr_basic(kernel_fn, train_fx, test_fx, train_x, train_input[1], test_x, test_input[1][test_mask], i*lr)


#             train_psnrs.append(train_psnr)
#             test_psnrs.append(test_psnr)
#             theories.append(theory)
#             pred = run_model(get_params(opt_state), ab, train_input[0])
#             errs.append(pred - train_input[1])
#             xs.append(i)
#     return get_params(opt_state), train_psnrs, test_psnrs, errs, np.array(theories), xs

# N_train = 32
# data_power = 1

# network_size = (4, 1024)

# learning_rate = 1e-5
# sgd_iters = 50001

# rand_key = random.PRNGKey(0)

# config.update('jax_disable_jit', False)

# # Signal
# M = 8
# N = N_train
# x_test = np.float32(np.linspace(0,1.,N*M,endpoint=False))
# x_train = x_test[::M]

# test_mask = np.ones(len(x_test), bool)
# test_mask[np.arange(0,x_test.shape[0],M)] = 0

# s = sample_random_powerlaw(rand_key, N*M, data_power) 
# s = (s-s.min()) / (s.max()-s.min()) - .5

# # Kernels
# bvals = np.float32(np.arange(1, N//2+1))
# ab_dict = {}
# # ab_dict = {r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0, 1]}
# ab_dict = {r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0, 0.5, 1, 1.5, 2]}
# ab_dict[r'$p = \infty$'] = (np.eye(bvals.shape[0])[0], bvals)
# ab_dict['No mapping'] = None


# # Train the networks

# rand_key, *ensemble_key = random.split(rand_key, 1 + len(ab_dict))

# outputs = {k : train_model(key, network_size, learning_rate, sgd_iters, 
#                            (x_train, s[::M]), (x_test, s), test_mask,
#                            optimizer=optimizers.sgd, ab=ab_dict[k], name=k) for k, key in zip(ab_dict, ensemble_key)}

# ab_dict.update({r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0.5, 1.5, 2]})

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']


# params = {'legend.fontsize': 24,
#          'axes.labelsize': 22,
#          'axes.titlesize': 26,
#          'xtick.labelsize':20,
#          'ytick.labelsize':20}
# pylab.rcParams.update(params)


# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['mathtext.rm'] = 'serif'

# plt.rcParams["font.family"] = "cmr10"
# names = ['$p = 0$', 
#          '$p = 0.5$',
#          '$p = 1$',
#          '$p = 1.5$',
#          '$p = 2$',
#          '$p = \\infty$']

# N_kernels = len(names)

# colors_k = np.array([[0.8872, 0.4281, 0.1875],
#     [0.8136, 0.6844, 0.0696],
#     [0.2634, 0.6634, 0.4134],
#     [0.0943, 0.5937, 0.8793],
#     [0.3936, 0.2946, 0.6330],
#     [0.7123, 0.2705, 0.3795]])
# linewidth = 3
# line_alpha = .8
# title_offset = -0.3

# xs = outputs[names[0]][-1]
# t_final = learning_rate * sgd_iters

# init_fn, apply_fn, kernel_fn = make_network(*network_size)
# run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, input_encoder(x, *ab))))

# fig3 = plt.figure(constrained_layout=True, figsize=(22,4))
# gs = fig3.add_gridspec(1, 4, width_ratios=[1,1,1.3,1.3])

# ### Plot NTK stuff

# H_rows = {k : compute_ntk(x_train, *ab_dict[k], kernel_fn) for k in names}

# samples = 100
# x_no_encoding = np.linspace(-np.pi, np.pi, samples)
# x_basic = np.stack([np.sin(x_no_encoding),np.cos(x_no_encoding)], axis=-1)
# relu_NTK = kernel_fn(x_no_encoding[:,None], x_no_encoding[:,None], 'ntk')
# basic_NTK = kernel_fn(x_basic, x_basic, 'ntk')

# ax = fig3.add_subplot(gs[0, 0])
# ax.imshow(relu_NTK, cmap='inferno', extent=[-.5,.5,.5,-.5])
# ax.xaxis.tick_top()
# extent = [-.5,.5]
# ax.set_xticks([-.5,.5])
# ax.set_yticks([-.5,.5])
# ax.set_xticklabels([fr'${t:g}$' for t in extent])
# ax.set_yticklabels([fr'${t:g}$' for t in extent])
# xtick = ax.get_xticks()
# ax.set_xticks(xtick)
# ax.set_xticklabels([fr'${t:g}$' for t in xtick])
# ax.set_title('(a) No mapping NTK', y=title_offset)

# ax = fig3.add_subplot(gs[0, 1])
# ax.imshow(basic_NTK, cmap='inferno', extent=[-.5,.5,.5,-.5])
# ax.xaxis.tick_top()
# ax.set_xticks([-.5,.5])
# ax.set_yticks([-.5,.5])
# ax.set_xticklabels([fr'${t:g}$' for t in extent])
# ax.set_yticklabels([fr'${t:g}$' for t in extent])
# ax.set_title('(b) Basic mapping NTK', y=title_offset)

# ax = fig3.add_subplot(gs[0, 2])
# for c, k in zip(colors_k, H_rows):
#   ntk_spatial = np.fft.fftshift(H_rows[k])
#   ax.plot(np.linspace(-.5, .5, 33, endpoint=True), np.append(ntk_spatial, ntk_spatial[0]), label=k, color=c, alpha=line_alpha, linewidth=linewidth)
# ax.set_title('(c) NTK spatial', y=title_offset)
# xtick = ax.get_xticks()
# ax.set_xticks(xtick)
# ax.set_xticklabels([fr'${t:g}$' for t in xtick])

# plt.grid(True, which='both', alpha=.3)
# plt.autoscale(enable=True, axis='x', tight=True)

# ax = fig3.add_subplot(gs[0, 3])
# for c, k in zip(colors_k, H_rows):
#   ntk_spectrum = 10**fplot(H_rows[k])
#   plt.semilogy(np.append(ntk_spectrum, ntk_spectrum[0]), label=k, color=c, alpha=line_alpha, linewidth=linewidth)
# ax.set_title('(d) NTK Fourier spectrum', y=title_offset)
# plt.xticks([0,8,16,24,32], ['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])

# plt.autoscale(enable=True, axis='x', tight=True)
# plt.grid(True, which='major', alpha=.3)
# plt.legend(loc='center left', bbox_to_anchor=(1,.5), handlelength=1)


# plt.savefig('1D_fig2.pdf', bbox_inches='tight', pad_inches=0)
# plt.show()
