import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class Generator(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.decoder = nn.Sequential(
			ResBlk(dim_in=config['content_depth'] + config['class_depth'], dim_out=512, normalize=True, upsample=False),
			# ResBlk(dim_in=512, dim_out=512, normalize=True, upsample=True),
			ResBlk(dim_in=512, dim_out=256, normalize=True, upsample=True),
			ResBlk(dim_in=256, dim_out=128, normalize=True, upsample=True),

			nn.InstanceNorm2d(num_features=128, affine=True),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0),

			nn.Sigmoid()
		)

	def forward(self, content_code, class_code):
		batch_size = content_code.shape[0]

		content_code = content_code.view((batch_size, -1, 16, 16))
		if self.training and self.config['content_std'] != 0:
			noise = torch.zeros_like(content_code)
			noise.normal_(mean=0, std=self.config['content_std'])

			content_code_regularized = content_code + noise
		else:
			content_code_regularized = content_code

		class_code = class_code.view((batch_size, -1, 16, 16))
		x = torch.cat((content_code_regularized, class_code), dim=1)

		return self.decoder(x)


class Discriminator(nn.Module):

	def __init__(self, config, max_conv_dim=512):
		super().__init__()

		self.config = config
		img_size = config['img_shape'][0]

		dim_in = 64 # 2**14 // img_size
		blocks = []
		blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

		repeat_num = int(np.log2(img_size)) - 2
		for _ in range(repeat_num):
			dim_out = min(dim_in*2, max_conv_dim)
			blocks += [ResBlk(dim_in, dim_out, downsample=True)]
			dim_in = dim_out

		# blocks += [nn.LeakyReLU(0.2)]
		# blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
		blocks += [nn.LeakyReLU(0.2)]
		blocks += [nn.Conv2d(dim_out, config['n_classes'], 4, 1, 0)]
		self.main = nn.Sequential(*blocks)

	def forward(self, x, y):
		out = self.main(x)
		out = out.view(out.size(0), -1)  # (batch, num_domains)
		idx = torch.LongTensor(range(y.size(0))).to(y.device)
		out = out[idx, y]  # (batch)
		return out


class ResBlk(nn.Module):

	def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False, upsample=False):
		super().__init__()

		self.actv = actv
		self.normalize = normalize
		self.downsample = downsample
		self.upsample = upsample
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
		if self.upsample:
			x = F.interpolate(x, scale_factor=2, mode='nearest')

		if self.learned_sc:
			x = self.conv1x1(x)

		if self.downsample:
			x = F.avg_pool2d(x, 2)

		return x

	def _residual(self, x):
		if self.normalize:
			x = self.norm1(x)

		x = self.actv(x)

		if self.upsample:
			x = F.interpolate(x, scale_factor=2, mode='nearest')

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
		return x / np.sqrt(2)  # unit variance


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
