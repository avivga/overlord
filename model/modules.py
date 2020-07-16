import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.class_embedding = nn.Embedding(num_embeddings=config['n_classes'], embedding_dim=config['class_dim'])

		self.adains = nn.Sequential(
			AdainResBlk(dim_in=config['content_depth'] + config['class_dim'], dim_out=512, style_dim=config['class_dim'], upsample=False),
			AdainResBlk(dim_in=512, dim_out=512, style_dim=config['class_dim'], upsample=False)
		)

		self.convs = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.ReflectionPad2d(padding=2),
			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1),
			nn.InstanceNorm2d(num_features=256),
			nn.ReLU(),

			nn.Upsample(scale_factor=2),
			nn.ReflectionPad2d(padding=2),
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
			nn.InstanceNorm2d(num_features=128),
			nn.ReLU(),

			nn.Upsample(scale_factor=2),
			nn.ReflectionPad2d(padding=2),
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1),
			nn.InstanceNorm2d(num_features=64),
			nn.ReLU(),

			nn.ReflectionPad2d(padding=3),
			nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1),
			nn.Tanh()
		)

		nn.init.uniform_(self.class_embedding.weight, a=-0.05, b=0.05)

	def forward(self, content_code, class_id):
		batch_size = content_code.shape[0]

		if self.training and self.config['content_std'] != 0:
			noise = torch.zeros_like(content_code)
			noise.normal_(mean=0, std=self.config['content_std'])

			content_code = content_code + noise

		class_code = self.class_embedding(class_id)
		class_code_repeated = class_code.view((batch_size, -1, 1, 1)).repeat((1, 1, 16, 16))
		x = torch.cat((content_code, class_code_repeated), dim=1)

		for block in self.adains:
			x = block(x, class_code)

		return self.convs(x)


class ContentEncoder(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.encoder = nn.Sequential(
			nn.ReflectionPad2d(padding=3),
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1),
			nn.InstanceNorm2d(num_features=64),
			nn.ReLU(),

			nn.ReflectionPad2d(padding=1),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
			nn.InstanceNorm2d(num_features=128),
			nn.ReLU(),

			nn.ReflectionPad2d(padding=1),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
			nn.InstanceNorm2d(num_features=256),
			nn.ReLU(),

			nn.ReflectionPad2d(padding=1),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2),
			nn.InstanceNorm2d(num_features=512),
			nn.ReLU(),

			ResBlk(dim_in=512, dim_out=512, normalize=True, downsample=False),
			ResBlk(dim_in=512, dim_out=config['content_depth'], normalize=True, downsample=False)
		)

	def forward(self, img):
		return self.encoder(img)


class Discriminator(nn.Module):

	def __init__(self, config, max_conv_dim=512):
		super().__init__()

		self.config = config
		img_size = config['img_shape'][0]

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
		blocks += [nn.Conv2d(dim_out, config['n_classes'], 1, 1, 0)]
		self.main = nn.Sequential(*blocks)

	def forward(self, x, y):
		out = self.main(x)
		out = out.view(out.size(0), -1)  # (batch, num_domains)
		idx = torch.LongTensor(range(y.size(0))).to(y.device)
		out = out[idx, y]  # (batch)
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
		return x / np.sqrt(2)  # unit variance


class AdaIN(nn.Module):

	def __init__(self, style_dim, num_features):
		super().__init__()

		self.norm = nn.InstanceNorm2d(num_features, affine=False)
		self.fc = nn.Linear(style_dim, num_features*2)

	def forward(self, x, s):
		h = self.fc(s)
		h = h.view(h.size(0), h.size(1), 1, 1)
		gamma, beta = torch.chunk(h, chunks=2, dim=1)
		return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):

	def __init__(self, dim_in, dim_out, style_dim, actv=nn.LeakyReLU(0.2), upsample=False):
		super().__init__()

		self.actv = actv
		self.upsample = upsample
		self.learned_sc = dim_in != dim_out
		self._build_weights(dim_in, dim_out, style_dim)

	def _build_weights(self, dim_in, dim_out, style_dim=64):
		self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
		self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
		self.norm1 = AdaIN(style_dim, dim_in)
		self.norm2 = AdaIN(style_dim, dim_out)
		if self.learned_sc:
			self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

	def _shortcut(self, x):
		if self.upsample:
			x = F.interpolate(x, scale_factor=2, mode='nearest')
		if self.learned_sc:
			x = self.conv1x1(x)
		return x

	def _residual(self, x, s):
		x = self.norm1(x, s)
		x = self.actv(x)
		if self.upsample:
			x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = self.conv1(x)
		x = self.norm2(x, s)
		x = self.actv(x)
		x = self.conv2(x)
		return x

	def forward(self, x, s):
		out = self._residual(x, s)
		out = (out + self._shortcut(x)) / np.sqrt(2)
		return out
