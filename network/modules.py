import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# stylegan2 modules
from model import ConstantInput, ToRGB, ModulatedConv2d, FusedLeakyReLU


class Generator(nn.Module):

	def __init__(self, img_size, content_dim, class_dim, style_dim, style_descriptor_dim):
		super().__init__()

		latent_dim = content_dim + class_dim + style_dim

		channel_multiplier = 2
		blur_kernel = [1, 3, 3, 1]

		self.channels = {
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256 * channel_multiplier,
			128: 128 * channel_multiplier,
			256: 64 * channel_multiplier,
			512: 32 * channel_multiplier,
			1024: 16 * channel_multiplier,
		}

		self.input = ConstantInput(self.channels[4])
		self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, latent_dim, blur_kernel=blur_kernel)
		self.to_rgb1 = ToRGB(self.channels[4], latent_dim, upsample=False)

		self.style_projector = nn.Linear(in_features=style_descriptor_dim, out_features=style_dim)

		self.log_size = int(math.log(img_size, 2))
		self.num_layers = (self.log_size - 2) * 2 + 1

		self.convs = nn.ModuleList()
		self.upsamples = nn.ModuleList()
		self.to_rgbs = nn.ModuleList()
		self.noises = nn.Module()

		in_channel = self.channels[4]

		for i in range(3, self.log_size + 1):
			out_channel = self.channels[2 ** i]

			self.convs.append(
				StyledConv(
					in_channel, out_channel,
					kernel_size=3, style_dim=latent_dim,
					upsample=True, blur_kernel=blur_kernel
				)
			)

			self.convs.append(
				StyledConv(
					out_channel, out_channel,
					kernel_size=3, style_dim=latent_dim,
					upsample=False, blur_kernel=blur_kernel
				)
			)

			self.to_rgbs.append(ToRGB(out_channel, latent_dim))

			in_channel = out_channel

		self.n_latent = self.log_size * 2 - 2

	def forward(self, content_code, class_code, style_descriptor):
		style_code = self.style_projector(style_descriptor)

		latent_code = torch.cat((content_code, class_code, style_code), dim=1)
		latent_code = latent_code.unsqueeze(dim=1).repeat(1, self.n_latent, 1)

		out = self.input(latent_code)
		out = self.conv1(out, latent_code[:, 0])

		skip = self.to_rgb1(out, latent_code[:, 1])

		i = 1
		for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
			out = conv1(out, latent_code[:, i])
			out = conv2(out, latent_code[:, i + 1])
			skip = to_rgb(out, latent_code[:, i + 2], skip)

			i += 2

		image = skip
		return image


class Encoder(nn.Module):

	def __init__(self, img_size, code_dim, dim_in=64, max_conv_dim=256):
		super().__init__()

		blocks = []
		blocks += [nn.Conv2d(in_channels=3, out_channels=dim_in, kernel_size=3, stride=1, padding=1)]

		repeat_num = int(np.log2(img_size)) - 2
		for _ in range(repeat_num):
			dim_out = min(dim_in*2, max_conv_dim)
			blocks += [ResBlk(dim_in, dim_out, downsample=True)]
			dim_in = dim_out

		blocks += [nn.LeakyReLU(0.2)]
		blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
		blocks += [nn.LeakyReLU(0.2)]
		blocks += [nn.Conv2d(dim_out, code_dim, 1, 1, 0)]

		self.main = nn.Sequential(*blocks)

	def forward(self, img):
		return torch.squeeze(self.main(img))


class Discriminator(nn.Module):

	def __init__(self, img_size, n_classes, max_conv_dim=256):
		super().__init__()

		dim_in = 2**14 // img_size
		blocks = []
		blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

		repeat_num = int(np.log2(img_size)) - 2
		for _ in range(repeat_num):
			dim_out = min(dim_in*2, max_conv_dim)
			blocks += [ResBlk(dim_in, dim_out, downsample=True)]
			dim_in = dim_out

		blocks += [nn.LeakyReLU(0.2)]
		blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
		blocks += [nn.LeakyReLU(0.2)]
		blocks += [nn.Conv2d(dim_out, n_classes, 1, 1, 0)]
		self.main = nn.Sequential(*blocks)

	def forward(self, x, y):
		out = self.main(x)
		out = out.view(out.size(0), -1)  # (batch, num_domains)
		idx = torch.LongTensor(range(y.size(0))).to(y.device)
		out = out[idx, y]  # (batch)
		return out


class StyledConv(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
		super().__init__()

		self.conv = ModulatedConv2d(
			in_channel,
			out_channel,
			kernel_size,
			style_dim,
			upsample=upsample,
			blur_kernel=blur_kernel,
			demodulate=demodulate,
		)

		# self.noise = NoiseInjection()
		# self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
		# self.activate = ScaledLeakyReLU(0.2)
		self.activate = FusedLeakyReLU(out_channel)

	def forward(self, input, style):
		out = self.conv(input, style)
		# out = self.noise(out, noise=noise)
		# out = out + self.bias
		out = self.activate(out)

		return out


class ResBlk(nn.Module):

	def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
		super().__init__()

		self.actv = actv
		self.normalize = normalize
		self.downsample = downsample
		self.learned_sc = dim_in != dim_out
		self._build_weights(dim_in, dim_out)

	def _build_weights(self, dim_in, dim_out):
		self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
		self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
		if self.normalize:
			self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
			self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
		if self.learned_sc:
			self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

	def _shortcut(self, x):
		if self.learned_sc:
			x = self.conv1x1(x)
		if self.downsample:
			x = F.avg_pool2d(x, 2)
		return x

	def _residual(self, x):
		if self.normalize:
			x = self.norm1(x)
		x = self.actv(x)
		x = self.conv1(x)
		if self.downsample:
			x = F.avg_pool2d(x, 2)
		if self.normalize:
			x = self.norm2(x)
		x = self.actv(x)
		x = self.conv2(x)
		return x

	def forward(self, x):
		x = self._shortcut(x) + self._residual(x)
		return x / math.sqrt(2)  # unit variance


class VGGFeatures(nn.Module):

	def __init__(self):
		super().__init__()

		self.vgg = models.vgg16(pretrained=True)

	def forward(self, x, layer_ids):
		output = []

		for i in range(layer_ids[-1] + 1):
			x = self.vgg.features[i](x)

			if i in layer_ids:
				output.append(x)

		return output


class VGGDistance(nn.Module):

	def __init__(self, vgg_features, layer_ids):
		super().__init__()

		self.vgg_features = vgg_features
		self.layer_ids = layer_ids

	def forward(self, I1, I2):
		batch_size = I1.size(0)

		f1 = self.vgg_features(I1, self.layer_ids)
		f2 = self.vgg_features(I2, self.layer_ids)

		loss = torch.abs(I1 - I2).view(batch_size, -1).mean(dim=1)

		for i in range(len(self.layer_ids)):
			layer_loss = torch.abs(f1[i] - f2[i]).view(batch_size, -1).mean(dim=1)
			loss = loss + layer_loss

		return loss.mean()


class VGGStyle(nn.Module):

	def __init__(self, vgg_features, layer_ids):
		super().__init__()

		self.vgg_features = vgg_features
		self.layer_ids = layer_ids

	def forward(self, x):
		features = self.vgg_features(x, self.layer_ids)

		means = []
		stds = []
		for i in range(len(self.layer_ids)):
			means_i = torch.mean(features[i], dim=[2, 3])
			stds_i = torch.std(features[i], dim=[2, 3])

			means.append(means_i)
			stds.append(stds_i)

		return torch.cat(means + stds, dim=1)
