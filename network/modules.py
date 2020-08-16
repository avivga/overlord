import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from model import (
	StyledConv,
	Blur,
	EqualLinear,
	EqualConv2d,
	ScaledLeakyReLU
)

from op import FusedLeakyReLU


class Generator(nn.Module):

	def __init__(self, config, channel=32, structure_channel=8, texture_channel=256, blur_kernel=(1, 3, 3, 1)):
		super().__init__()

		self.config = config

		ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
		upsample = (False, False, False, False, True, True, True, True)

		self.layers = nn.ModuleList()
		in_ch = structure_channel
		for ch_mul, up in zip(ch_multiplier, upsample):
			self.layers.append(StyledResBlock(in_ch, channel * ch_mul, texture_channel, up, blur_kernel))
			in_ch = channel * ch_mul

		self.to_rgb = nn.Sequential(
			ConvLayer(in_ch, 3, 1, activate=False),
			nn.Sigmoid()
		)

	def forward(self, content_code, class_code):
		batch_size = content_code.shape[0]

		content_code = content_code.view((batch_size, -1, 4, 4))
		if self.training and self.config['content_std'] != 0:
			noise = torch.zeros_like(content_code)
			noise.normal_(mean=0, std=self.config['content_std'])

			out = content_code + noise
		else:
			out = content_code

		for layer in self.layers:
			out = layer(out, class_code, None)

		out = self.to_rgb(out)
		return out


# class Generator(nn.Module):
#
# 	def __init__(self, config):
# 		super().__init__()
#
# 		self.config = config
#
# 		self.decoder = nn.Sequential(
# 			ResBlk(dim_in=config['content_depth'] + config['class_depth'], dim_out=512, normalize=True, upsample=False),
# 			# ResBlk(dim_in=512, dim_out=512, normalize=True, upsample=True),
# 			ResBlk(dim_in=512, dim_out=256, normalize=True, upsample=True),
# 			ResBlk(dim_in=256, dim_out=128, normalize=True, upsample=True),
#
# 			nn.InstanceNorm2d(num_features=128, affine=True),
# 			nn.LeakyReLU(negative_slope=0.2),
# 			nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0),
#
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, content_code, class_code):
# 		batch_size = content_code.shape[0]
#
# 		content_code = content_code.view((batch_size, -1, 16, 16))
# 		if self.training and self.config['content_std'] != 0:
# 			noise = torch.zeros_like(content_code)
# 			noise.normal_(mean=0, std=self.config['content_std'])
#
# 			content_code_regularized = content_code + noise
# 		else:
# 			content_code_regularized = content_code
#
# 		class_code = class_code.view((batch_size, -1, 16, 16))
# 		x = torch.cat((content_code_regularized, class_code), dim=1)
#
# 		return self.decoder(x)


class EqualConvTranspose2d(nn.Module):
	def __init__(
			self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
	):
		super().__init__()

		self.weight = nn.Parameter(
			torch.randn(in_channel, out_channel, kernel_size, kernel_size)
		)
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))

		else:
			self.bias = None

	def forward(self, input):
		out = F.conv_transpose2d(
			input,
			self.weight * self.scale,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
			)

		return out

	def __repr__(self):
		return (
			f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
			f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
		)


class ConvLayer(nn.Sequential):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			upsample=False,
			downsample=False,
			blur_kernel=(1, 3, 3, 1),
			bias=True,
			activate=True,
			padding="zero",
	):
		layers = []

		self.padding = 0
		stride = 1

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2

		if upsample:
			layers.append(
				EqualConvTranspose2d(
					in_channel,
					out_channel,
					kernel_size,
					padding=0,
					stride=2,
					bias=bias and not activate,
				)
			)

			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2 + factor - 1
			pad1 = p // 2 + 1

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

		else:
			if not downsample:
				if padding == "zero":
					self.padding = (kernel_size - 1) // 2

				elif padding == "reflect":
					padding = (kernel_size - 1) // 2

					if padding > 0:
						layers.append(nn.ReflectionPad2d(padding))

					self.padding = 0

				elif padding != "valid":
					raise ValueError('Padding should be "zero", "reflect", or "valid"')

			layers.append(
				EqualConv2d(
					in_channel,
					out_channel,
					kernel_size,
					padding=self.padding,
					stride=stride,
					bias=bias and not activate,
				)
			)

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))

			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class StyledResBlock(nn.Module):
	def __init__(
			self, in_channel, out_channel, style_dim, upsample, blur_kernel=(1, 3, 3, 1)
	):
		super().__init__()

		self.conv1 = StyledConv(
			in_channel,
			out_channel,
			3,
			style_dim,
			upsample=upsample,
			blur_kernel=blur_kernel,
		)

		self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim)

		if upsample or in_channel != out_channel:
			self.skip = ConvLayer(
				in_channel,
				out_channel,
				1,
				upsample=upsample,
				blur_kernel=blur_kernel,
				bias=False,
				activate=False,
			)

		else:
			self.skip = None

	def forward(self, input, style, noise=None):
		out = self.conv1(input, style, noise)
		out = self.conv2(out, style, noise)

		if self.skip is not None:
			skip = self.skip(input)

		else:
			skip = input

		return (out + skip) / math.sqrt(2)


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
