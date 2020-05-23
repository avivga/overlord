import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from model.utils import he_init


class Generator(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.content_embedding = nn.Embedding(config['n_imgs'], config['content_dim'])
		# self.style_embedding = nn.Embedding(config['n_imgs'], config['style_dim'])
		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

		self.class_style_modulation = nn.Sequential(
			nn.Linear(in_features=config['class_dim'] + config['style_dim'], out_features=config['style_dim'] * 2),
			nn.LeakyReLU(negative_slope=0.2),

			nn.Linear(in_features=config['style_dim'] * 2, out_features=config['style_dim']),
			nn.LeakyReLU(negative_slope=0.2)
		)

		self.modulation = Modulation(config['style_dim'], n_adain_layers=4, adain_dim=256)
		self.decoder = Decoder(config['content_dim'], n_adain_layers=4, adain_dim=256, img_shape=config['img_shape'])

		self.apply(self.weights_init)

	def forward(self, content_img_id, style_code, class_id):
		batch_size = content_img_id.shape[0]

		content_code = self.content_embedding(content_img_id)
		class_code = self.class_embedding(class_id)

		if self.training and self.config['content_std'] != 0:
			noise = torch.zeros_like(content_code)
			noise.normal_(mean=0, std=self.config['content_std'])

			regularized_content_code = content_code + noise
		else:
			regularized_content_code = content_code

		if self.training and self.config['style_std'] != 0:
			noise = torch.zeros_like(style_code)
			noise.normal_(mean=0, std=self.config['style_std'])

			regularized_style_code = style_code + noise
		else:
			regularized_style_code = style_code

		class_with_style_code = torch.cat((class_code, regularized_style_code), dim=1)
		class_with_style_code = self.class_style_modulation(class_with_style_code)

		adain_params = self.modulation(class_with_style_code)
		generated_img = self.decoder(regularized_content_code, adain_params)

		return {
			'img': generated_img,
			'content_code': content_code,
			'style_code': style_code
		}

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class Modulation(nn.Module):

	def __init__(self, code_dim, n_adain_layers, adain_dim):
		super().__init__()

		self.__n_adain_layers = n_adain_layers
		self.__adain_dim = adain_dim

		self.adain_per_layer = nn.ModuleList([
			nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
			for _ in range(n_adain_layers)
		])

	def forward(self, x):
		adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
		adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

		return adain_params


class Decoder(nn.Module):

	def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
		super().__init__()

		self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
		self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
		self.__adain_dim = adain_dim

		self.fc_layers = nn.Sequential(
			nn.Linear(
				in_features=content_dim,
				out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
			),

			nn.LeakyReLU(negative_slope=0.2),

			nn.Linear(
				in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
				out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
			),

			nn.LeakyReLU(negative_slope=0.2),

			nn.Linear(
				in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
				out_features=self.__initial_height * self.__initial_width * adain_dim
			),

			nn.LeakyReLU(negative_slope=0.2),
		)

		self.adain_conv_layers = nn.ModuleList()
		for i in range(n_adain_layers):
			self.adain_conv_layers += [
				nn.Upsample(scale_factor=(2, 2)),
				nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
				nn.LeakyReLU(negative_slope=0.2),
				AdaptiveInstanceNorm2d(adain_layer_idx=i)
			]

		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

		self.last_conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
			nn.LeakyReLU(negative_slope=0.2),

			nn.Conv2d(in_channels=64, out_channels=img_shape[-1], padding=3, kernel_size=7),
			nn.Sigmoid()
		)

	def assign_adain_params(self, adain_params):
		for m in self.adain_conv_layers.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				m.bias = adain_params[:, m.adain_layer_idx, :, 0]
				m.weight = adain_params[:, m.adain_layer_idx, :, 1]

	def forward(self, content_code, class_adain_params):
		self.assign_adain_params(class_adain_params)

		x = self.fc_layers(content_code)
		x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
		x = self.adain_conv_layers(x)
		x = self.last_conv_layers(x)

		return x


class AdaptiveInstanceNorm2d(nn.Module):

	def __init__(self, adain_layer_idx):
		super().__init__()
		self.weight = None
		self.bias = None
		self.adain_layer_idx = adain_layer_idx

	def forward(self, x):
		b, c = x.shape[0], x.shape[1]

		x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
		weight = self.weight.contiguous().view(-1)
		bias = self.bias.contiguous().view(-1)

		out = F.batch_norm(
			x_reshaped, running_mean=None, running_var=None,
			weight=weight, bias=bias, training=True
		)

		out = out.view(b, c, *x.shape[2:])
		return out


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

		self.apply(he_init)

	def forward(self, x, y):
		out = self.main(x)
		out = out.view(out.size(0), -1)  # (batch, num_domains)
		idx = torch.LongTensor(range(y.size(0))).to(y.device)
		out = out[idx, y]  # (batch)
		return out


class StyleEncoder(nn.Module):

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
		self.shared = nn.Sequential(*blocks)

		self.unshared = nn.ModuleList()
		for _ in range(config['n_classes']):
			self.unshared += [nn.Linear(dim_out, config['style_dim'])]

		self.apply(he_init)

	def forward(self, x, y):
		h = self.shared(x)
		h = h.view(h.size(0), -1)
		out = []
		for layer in self.unshared:
			out += [layer(h)]
		out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
		idx = torch.LongTensor(range(y.size(0))).to(y.device)
		s = out[idx, y]  # (batch, style_dim)
		return s


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
