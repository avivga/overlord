import math
import numpy as np

import torch
from torch import nn
from torchvision import models

from model import ConstantInput, ToRGB, ModulatedConv2d, FusedLeakyReLU


class Generator(nn.Module):

	def __init__(self, size, style_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.size = size
		self.style_dim = style_dim

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
		self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
		self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

		self.log_size = int(math.log(size, 2))
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
					in_channel,
					out_channel,
					3,
					style_dim,
					upsample=True,
					blur_kernel=blur_kernel,
				)
			)

			self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
			self.to_rgbs.append(ToRGB(out_channel, style_dim))

			in_channel = out_channel

		self.n_latent = self.log_size * 2 - 2

	def forward(self, content_codes, class_codes):
		styles = torch.cat((content_codes, class_codes), dim=1)
		latent = styles.unsqueeze(dim=1).repeat(1, self.n_latent, 1)
		# latent = styles.view((-1, self.n_latent, 512))

		out = self.input(latent)
		out = self.conv1(out, latent[:, 0])

		skip = self.to_rgb1(out, latent[:, 1])

		i = 1
		for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
			out = conv1(out, latent[:, i])
			out = conv2(out, latent[:, i + 1])
			skip = to_rgb(out, latent[:, i + 2], skip)

			i += 2

		image = skip
		return image


class StyledConv(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			style_dim,
			upsample=False,
			blur_kernel=[1, 3, 3, 1],
			demodulate=True,
	):
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


# class Generator(nn.Module):
#
# 	def __init__(self, config, channel=32, structure_channel=8, texture_channel=256, blur_kernel=(1, 3, 3, 1)):
# 		super().__init__()
#
# 		self.config = config
#
# 		ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
# 		upsample = (False, False, False, False, True, True, True, True)
#
# 		self.layers = nn.ModuleList()
# 		in_ch = structure_channel
# 		for ch_mul, up in zip(ch_multiplier, upsample):
# 			self.layers.append(StyledResBlock(in_ch, channel * ch_mul, texture_channel, up, blur_kernel))
# 			in_ch = channel * ch_mul
#
# 		self.to_rgb = nn.Sequential(
# 			ConvLayer(in_ch, 3, 1, activate=False),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, content_code, class_code):
# 		batch_size = content_code.shape[0]
#
# 		content_code = content_code.view((batch_size, -1, 4, 4))
# 		if self.training and self.config['content_std'] != 0:
# 			noise = torch.zeros_like(content_code)
# 			noise.normal_(mean=0, std=self.config['content_std'])
#
# 			out = content_code + noise
# 		else:
# 			out = content_code
#
# 		for layer in self.layers:
# 			out = layer(out, class_code, None)
#
# 		out = self.to_rgb(out)
# 		return out


class NetVGGFeatures(nn.Module):

	def __init__(self, layer_ids):
		super().__init__()

		self.vggnet = models.vgg16(pretrained=True)
		self.layer_ids = layer_ids

	def forward(self, x):
		output = []
		for i in range(self.layer_ids[-1] + 1):
			x = self.vggnet.features[i](x)

			if i in self.layer_ids:
				output.append(x)

		return output


class VGGDistance(nn.Module):

	def __init__(self, layer_ids):
		super().__init__()

		self.vgg = NetVGGFeatures(layer_ids)
		self.layer_ids = layer_ids

	def forward(self, I1, I2):
		b_sz = I1.size(0)
		f1 = self.vgg(I1)
		f2 = self.vgg(I2)

		loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

		for i in range(len(self.layer_ids)):
			layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
			loss = loss + layer_loss

		return loss.mean()
