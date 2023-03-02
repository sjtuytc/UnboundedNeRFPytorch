import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import pdb
import time
import torch
# import random
# from jax import random
import torch.nn as nn
import numpy as np
from scipy.special import jv
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import axes3d


class VoxelGrid(nn.Module):
    # the V o x e l G ri d operator
    def __init__(self, grid_len=1000, data_point_num=100):
        super(VoxelGrid, self).__init__()
        self.grid_len = grid_len
        self.data_point_num = data_point_num
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
        jacobian_y_w = np.zeros((self.data_point_num, self.grid_len))
        for idx in range(self.data_point_num):
            real_x = idx / self.data_point_num
            left_grid = int(real_x // (1 / self.grid_len))
            right_grid = left_grid + 1
            if left_grid >= 0:
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


class FourierGrid(nn.Module):
    # the FourierGrid operator
    def __init__(self, grid_len=1000, band_num=10, data_point_num=100):
        super(FourierGrid, self).__init__()
        self.grid_len = grid_len
        self.data_point_num = data_point_num
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
        jacobian_y_w = np.zeros((self.data_point_num, self.grid_len * self.band_num))
        for idx in range(self.data_point_num):  # for all data points
            real_x = idx / self.data_point_num   # the real x value
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


# build models and train them
def train_model(one_model):
    # training and testing VoxelGrid
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


def get_fg_gtk_spectrum_by_band_num(band_num):
    test_fg = FourierGrid(grid_len=grid_len, band_num=band_num * 2)
    fg_gtk = test_fg()
    # fg_gtk = (fg_gtk - fg_gtk.min()) / (fg_gtk.max() - fg_gtk.min())
    fg_gtk_spectrum = 10**fplot(fg_gtk)
    fg_plot = gaussian_filter1d(fg_gtk_spectrum[0], sigma=2)
    return fg_plot


# hyperparameters
title_offset = -0.29
bbox_offset = 1.44
data_point_num = 100
grid_len = 10
freq_num = 10
colors_k = np.array([[0.8872, 0.4281, 0.1875],
    [0.8136, 0.6844, 0.0696],
    [0.2634, 0.6634, 0.4134],
    [0.0943, 0.5937, 0.8793],
    [0.3936, 0.2946, 0.6330],
    [0.7123, 0.2705, 0.3795]])
linewidth = 1.0
line_alpha = .8
title_font_size = 7.4
legend_font_size = 6
label_size = 7
# matplotlib.rcParams["font.family"] = 'Arial'
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size 

# begin plot 
fig3 = plt.figure(constrained_layout=True, figsize=(4, 4))
gs = fig3.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
# 100 * 100 datapoints, 10*10 params (grid_len=10)
test_vg = VoxelGrid(grid_len=grid_len * freq_num)
vg_gtk = test_vg()
vg_gtk_normalized = (vg_gtk - vg_gtk.min()) / (vg_gtk.max() - vg_gtk.min())
ax = fig3.add_subplot(gs[0, 0])
ax.imshow(vg_gtk_normalized)
ax.set_xticks([*range(0, 100, 20)] + [100])
ax.set_yticks([*range(0, 100, 20)] + [100])
ax.grid(linestyle = '--', linewidth = 0.3)
ax.set_title('(a) VoxelGrid GTK', y=title_offset, fontsize=title_font_size)

ax = fig3.add_subplot(gs[0, 1])
test_fg = FourierGrid(grid_len=grid_len, band_num=freq_num)
fg_gtk = test_fg()
fg_gtk = (fg_gtk - fg_gtk.min()) / (fg_gtk.max() - fg_gtk.min())
ax.imshow(fg_gtk)
ax.set_xticks([*range(0, 100, 20)] + [100])
ax.set_yticks([*range(0, 100, 20)] + [100])
ax.grid(linestyle = '--', linewidth = 0.3)
ax.set_title('(b) FourierGrid GTK', y=title_offset, fontsize=title_font_size)

ax = fig3.add_subplot(gs[1, 0])
w_vg, v_vg = np.linalg.eig(vg_gtk)
w_fg, v_fg = np.linalg.eig(fg_gtk)
fplot = lambda x : np.fft.fftshift(np.log10(np.abs(np.fft.fft(x))))
vg_gtk_spectrum = 10**fplot(vg_gtk)
vg_plot = gaussian_filter1d(vg_gtk_spectrum[0], sigma=2)

fg_gtk_plot_1 = get_fg_gtk_spectrum_by_band_num(band_num=1)
fg_gtk_plot_5 = get_fg_gtk_spectrum_by_band_num(band_num=5)
fg_gtk_plot_10 = get_fg_gtk_spectrum_by_band_num(band_num=10)
plt.autoscale(enable=True, axis='x', tight=True)
# plt.plot(vg_plot, label='VoxelGrid', color=colors_k[0], alpha=line_alpha, linewidth=linewidth)
plt.semilogy(np.append(vg_plot, vg_plot[0]), label='VoxelGrid', color=colors_k[0], alpha=line_alpha, linewidth=linewidth)
# plt.semilogy(fg_gtk_plot_1, label='FourierGrid (l=1)', color=colors_k[2], alpha=line_alpha, linewidth=linewidth)
plt.semilogy(np.append(fg_gtk_plot_1, fg_gtk_plot_1[0]), label='FourierGrid (l=1)', color=colors_k[2], alpha=line_alpha, linewidth=linewidth)
# plt.semilogy(fg_gtk_plot_5, label='FourierGrid (l=5)', color=colors_k[3], alpha=line_alpha, linewidth=linewidth)
plt.semilogy(np.append(fg_gtk_plot_5, fg_gtk_plot_5[0]), label='FourierGrid (l=5)', color=colors_k[3], alpha=line_alpha, linewidth=linewidth)
# plt.semilogy(fg_gtk_plot_10, label='FourierGrid (l=10)', color=colors_k[4], alpha=line_alpha, linewidth=linewidth)
plt.semilogy(np.append(fg_gtk_plot_10, fg_gtk_plot_10[0]), label='FourierGrid (l=10)', color=colors_k[4], alpha=line_alpha, linewidth=linewidth)
plt.xticks([0,25,50,75,100], ['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
ax.set_yticks([0.1, 1, 10, 100])
ax.legend(loc='upper left', bbox_to_anchor=(-0.01, bbox_offset), handlelength=1, fontsize=legend_font_size, fancybox=False, ncol=1)
ax.set_title('(c) GTK Fourier Spectrum', y=title_offset, fontsize=title_font_size)


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


def get_sine_signal():
    return np.array([np.sin(x / (train_num*sample_interval) * 2 * np.pi) for x in range(train_num*sample_interval)])


def get_bessel_signal():
    # return np.array([np.exp(x / train_num*sample_interval) for x in range(train_num*sample_interval)])
    return np.array([jv(1, x / 4) for x in range(train_num*sample_interval)])

##  Fitting experiments
# hyperparameters
rand_key = np.array([0, 0], dtype=np.uint32)
train_num = 7
sample_interval = 4
data_power = 0.5
lr = 1


# setup data
x_test = np.float32(np.linspace(0, 1., train_num * sample_interval, endpoint=False))
x_train = x_test[0:len(x_test):sample_interval]

# signal = get_sine_signal()
signal = get_bessel_signal()
signal = (signal-signal.min()) / (signal.max()-signal.min())

y_train = signal[0:len(x_test):sample_interval]
y_test_gt = signal


freq_num = 3
test_vg_small = VoxelGrid(grid_len=10 * freq_num)
test_vg_large = VoxelGrid(grid_len=100 * freq_num)
test_fg_small = FourierGrid(grid_len=10, band_num=freq_num)
test_fg_large = FourierGrid(grid_len=100, band_num=freq_num)
train_loss, test_loss, test_y_vg_small = train_model(test_vg_small)
train_loss, test_loss, test_y_fg_small = train_model(test_fg_small)

ax = fig3.add_subplot(gs[1, 1])
ax.plot(x_test, signal, label='Target signal', color='k', linewidth=1, alpha=line_alpha, zorder=1)
ax.scatter(x_train, y_train, color='w', edgecolors='k', linewidths=1, s=20, linewidth=1, label='Training points', zorder=2)
ax.plot(x_test, test_y_vg_small, label='Learned by VoxelGrid', color=colors_k[0], linewidth=1, alpha=line_alpha, zorder=1)
ax.plot(x_test, test_y_fg_small, label='Learned by FourierGrid', color=colors_k[3], linewidth=1, alpha=line_alpha, zorder=1)
ax.set_title('(d) 1D Regression', y=title_offset, fontsize=title_font_size)
ax.set_xticks(np.linspace(0.0, 1.0, num=5, endpoint=True))
ax.legend(loc='upper left', bbox_to_anchor=(-0.01, bbox_offset), handlelength=1, fontsize=legend_font_size, fancybox=False, ncol=1)

print("Plotting figures 1")
plt.savefig("figures/vg_fg_gtk.jpg", dpi=300) # for example
plt.savefig("figures/vg_fg_gtk.pdf", format="pdf")

####################################################################
# plotting a new diagram, figure 2
####################################################################
def calculate_Delta(gtk, y1_data, y2_data):
    # batch_y: [1 * 2 * B]
    batch_y_transpose = np.expand_dims(np.stack([y1_data, y2_data]).reshape(2, -1).transpose(), axis=2)
    batch_y = np.expand_dims(np.stack([y1_data, y2_data]).reshape(2, -1).transpose(), axis=1)
    batch_size = batch_y.shape[0]
    batch_gtk = np.array([np.linalg.inv(gtk) for i in range(batch_size)])
    result = np.matmul(batch_y, np.matmul(batch_gtk, batch_y_transpose)).squeeze()
    return result


one_vg = VoxelGrid(grid_len=grid_len, data_point_num=2)
vg_gtk = one_vg()
one_fg = FourierGrid(grid_len=grid_len, data_point_num=2)
fg_gtk = one_fg()
y1_data, y2_data, Z = axes3d.get_test_data(0.05)
y1_data = y1_data / y1_data.max()
y2_data = y2_data / y2_data.max()
vg_values = calculate_Delta(vg_gtk, y1_data, y2_data).reshape(Z.shape)
vg_values /= vg_values.max()
fg_values = calculate_Delta(fg_gtk, y1_data, y2_data).reshape(Z.shape) 
fg_values /= fg_values.max()

# begin plot 
fig4 = plt.figure(constrained_layout=True, figsize=(4, 2))
gs = fig4.add_gridspec(1, 2, width_ratios=[1.04, 0.96])
# 100 * 100 datapoints, 10*10 params (grid_len=10)
ax = fig4.add_subplot(gs[0, 0])

X=np.array([0.15, 0.44, 0.3, 0.56, 0.78, 0.72])
Y=np.array([0.18, 0.34, 0.51, 0.72, 0.81, 0.93])
annotations=["Synthetic-NeRF", "NSVF", "BlendedMVS", "UT&T", "M360", "SFMB"]

ax.scatter(X[:3], Y[:3], s=30, color=colors_k[0], marker="s")
ax.scatter(X[3:], Y[3:], s=30, color=colors_k[3], marker="o")
ax.annotate(annotations[0], (X[0], Y[0]), fontsize=title_font_size)
ax.annotate(annotations[1], (X[1], Y[1]), fontsize=title_font_size)
ax.annotate(annotations[2], (X[2], Y[2]), fontsize=title_font_size)
ax.annotate(annotations[3], (X[3], Y[3]), fontsize=title_font_size)
ax.annotate(annotations[4], (X[4], Y[4]), fontsize=title_font_size)
ax.annotate(annotations[5], (X[5], Y[5]), fontsize=title_font_size)
ax.set_xlabel("Density Norms", size=title_font_size)
ax.set_ylabel("Generalization Gap", size=title_font_size)

ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(linestyle='--', linewidth = 0.3)
ax.set_title('(a) Bounded vs. Unbounded Scenes',  y=-0.4, fontsize=title_font_size)


ax = fig4.add_subplot(gs[0, 1])
diffrences = vg_values - fg_values
ax.imshow(diffrences, cmap='coolwarm', )
ax.set_xticks([0,24,48,72,96, 120], [-1,-0.6,-0.2,0.2,0.6,1])
ax.set_yticks([0,24,48,72,96, 120], [-1,-0.6,-0.2,0.2,0.6,1])
ax.grid(linestyle = '--', linewidth = 0.3)
ax.set_xlabel("y1", size=title_font_size, labelpad=1)
ax.set_ylabel("y2", size=title_font_size, labelpad=-1)
ax.set_title('(b) Generalization Bound Diff.',  y=-0.491, fontsize=title_font_size)
plot = ax.pcolor(diffrences)
plt.colorbar(plot)

print("Plotting figures 2")
plt.savefig("figures/unbounded.jpg", dpi=300) # for example
plt.savefig("figures/unbounded.pdf", format="pdf")
pdb.set_trace()


# ax = fig3.add_subplot(gs[0, 1])
# test_fg = FourierGrid(grid_len=grid_len, band_num=freq_num)
# fg_gtk = test_fg()
# fg_gtk = (fg_gtk - fg_gtk.min()) / (fg_gtk.max() - fg_gtk.min())
# ax.imshow(fg_gtk)
# ax.set_xticks([*range(0, 100, 20)] + [100])
# ax.set_yticks([*range(0, 100, 20)] + [100])
# ax.grid(linestyle = '--', linewidth = 0.3)
# ax.set_title('(b) FourierGrid GTK', y=title_offset, fontsize=title_font_size)

pdb.set_trace()

######################################################################
# 3D plotting
######################################################################

ax = plt.figure(constrained_layout=True).add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(y1_data, y2_data, vg_values, color=colors_k[0], edgecolor=colors_k[0], lw=0.5, rstride=8, cstride=8,
                alpha=0.3)
ax.plot_surface(y1_data, y2_data, fg_values, color=colors_k[3], edgecolor=colors_k[3], lw=0.5, rstride=8, cstride=8,
                alpha=0.3)
# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
color_shift = 0.9
ax.contourf(y1_data, y2_data, fg_values - vg_values + color_shift, zdir='z', offset=0.0, cmap='coolwarm', vmin=0.2, vmax=1.6)
# ax.contourf(y1_data, y2_data, Z, zdir='x', offset=-40, cmap='coolwarm')
# ax.contourf(y1_data, y2_data, Z, zdir='y', offset=40, cmap='coolwarm')
ax.margins(x=0, y=0, z=0)

ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1),
       xlabel='y1', ylabel='y2', zlabel='Gen. Bound')

print("Plotting figures 3, supplementary")
plt.savefig("figures/compare_generalization.jpg", dpi=300) # for example
plt.savefig("figures/compare_generalization.pdf", format="pdf")

