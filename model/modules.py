import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.content_embedding = nn.Embedding(config['n_imgs'], config['content_dim'], config['content_std'])
		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
		self.modulation = Modulation(config['class_dim'], config['generator']['n_adain_layers'], config['generator']['adain_dim'])
		self.decoder = Decoder(config['content_dim'], config['generator']['n_adain_layers'], config['generator']['adain_dim'], config['img_shape'])

	def forward(self, img_id, class_id):
		content_code = self.content_embedding(img_id)

		if self.training and self.config['content_std'] != 0:
			noise = torch.zeros_like(content_code)
			noise.normal_(mean=0, std=self.config['content_std'])

			regularized_content_code = content_code + noise
		else:
			regularized_content_code = content_code

		class_code = self.class_embedding(class_id)
		class_adain_params = self.modulation(class_code)
		generated_img = self.decoder(regularized_content_code, class_adain_params)

		return {
			'img': generated_img,
			'content_code': content_code,
			'class_code': class_code
		}

	def init(self):
		self.apply(self.weights_init)

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

			nn.LeakyReLU(),

			nn.Linear(
				in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
				out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
			),

			nn.LeakyReLU(),

			nn.Linear(
				in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
				out_features=self.__initial_height * self.__initial_width * adain_dim
			),

			nn.LeakyReLU()
		)

		self.adain_conv_layers = nn.ModuleList()
		for i in range(n_adain_layers):
			self.adain_conv_layers += [
				nn.Upsample(scale_factor=(2, 2)),
				nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
				nn.LeakyReLU(),
				AdaptiveInstanceNorm2d(adain_layer_idx=i)
			]

		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

		self.last_conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels=64, out_channels=img_shape[2], padding=3, kernel_size=7),
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

	def __init__(self, config):
		super().__init__()

		self.config = config

		layers = []
		for i in range(self.config['discriminator']['n_layers']):
			in_channels = self.config['discriminator']['filters'] * (2 ** (i - 1)) if i > 0 else 3
			out_channels = 2 * in_channels if i > 0 else self.config['discriminator']['filters']

			layers += [
				nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
				nn.LeakyReLU(negative_slope=0.2, inplace=True)
			]

		self.convs = nn.Sequential(*layers)

		self.linear = nn.Linear(
			in_features=self.config['discriminator']['filters'] * (2 ** (self.config['discriminator']['n_layers'] - 1)),
			out_features=1
		)

		self.class_embedding = nn.Embedding(
			num_embeddings=config['n_classes'],
			embedding_dim=self.config['discriminator']['filters'] * (2 ** (self.config['discriminator']['n_layers'] - 1))
		)

	def forward(self, img, class_id):
		x = self.convs(img)
		h = torch.sum(x, dim=[2, 3])

		out = self.linear(h)
		return out + torch.sum(self.class_embedding(class_id) * h, dim=1, keepdim=True)
