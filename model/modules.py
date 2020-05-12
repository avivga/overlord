import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.content_embedding = nn.Embedding(config['n_imgs'], config['content_dim'], config['content_std'])
		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		self.decoder = Decoder(config['content_dim'], config['n_adain_layers'], config['adain_dim'], config['img_shape'])

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


class Conv2dBlock(nn.Module):

	def __init__(self, input_dim ,output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
		super().__init__()

		self.use_bias = True
		# initialize padding
		if pad_type == 'reflect':
			self.pad = nn.ReflectionPad2d(padding)
		elif pad_type == 'replicate':
			self.pad = nn.ReplicationPad2d(padding)
		elif pad_type == 'zero':
			self.pad = nn.ZeroPad2d(padding)
		else:
			assert 0, "Unsupported padding type: {}".format(pad_type)

		# initialize normalization
		norm_dim = output_dim
		if norm == 'bn':
			self.norm = nn.BatchNorm2d(norm_dim)
		elif norm == 'in':
			#self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
			self.norm = nn.InstanceNorm2d(norm_dim)
		# elif norm == 'ln':
		# 	self.norm = LayerNorm(norm_dim)
		elif norm == 'adain':
			self.norm = AdaptiveInstanceNorm2d(norm_dim)
		elif norm == 'none' or norm == 'sn':
			self.norm = None
		else:
			assert 0, "Unsupported normalization: {}".format(norm)

		# initialize activation
		if activation == 'relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation == 'lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation == 'prelu':
			self.activation = nn.PReLU()
		elif activation == 'selu':
			self.activation = nn.SELU(inplace=True)
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		elif activation == 'none':
			self.activation = None
		else:
			assert 0, "Unsupported activation: {}".format(activation)

		# initialize convolution
		if norm == 'sn':
			self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
		else:
			self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

	def forward(self, x):
		x = self.conv(self.pad(x))
		if self.norm:
			x = self.norm(x)
		if self.activation:
			x = self.activation(x)
		return x


def l2normalize(v, eps=1e-12):
	return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

	def __init__(self, module, name='weight', power_iterations=1):
		super(SpectralNorm, self).__init__()
		self.module = module
		self.name = name
		self.power_iterations = power_iterations
		if not self._made_params():
			self._make_params()

	def _update_u_v(self):
		u = getattr(self.module, self.name + "_u")
		v = getattr(self.module, self.name + "_v")
		w = getattr(self.module, self.name + "_bar")

		height = w.data.shape[0]
		for _ in range(self.power_iterations):
			v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
			u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

		# sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
		sigma = u.dot(w.view(height, -1).mv(v))
		setattr(self.module, self.name, w / sigma.expand_as(w))

	def _made_params(self):
		try:
			u = getattr(self.module, self.name + "_u")
			v = getattr(self.module, self.name + "_v")
			w = getattr(self.module, self.name + "_bar")
			return True
		except AttributeError:
			return False

	def _make_params(self):
		w = getattr(self.module, self.name)

		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]

		u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
		u.data = l2normalize(u.data)
		v.data = l2normalize(v.data)
		w_bar = nn.Parameter(w.data)

		del self.module._parameters[self.name]

		self.module.register_parameter(self.name + "_u", u)
		self.module.register_parameter(self.name + "_v", v)
		self.module.register_parameter(self.name + "_bar", w_bar)

	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)


class MsImageDis(nn.Module):

	def __init__(self):
		super().__init__()

		self.n_layer = 4
		self.gan_type = 'lsgan'
		self.dim = 64
		self.norm = 'none'
		self.activ = 'lrelu'
		self.num_scales = 3
		self.pad_type = 'reflect'
		self.input_dim = 3
		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
		self.cnns = nn.ModuleList()
		for _ in range(self.num_scales):
			self.cnns.append(self._make_net())

	def _make_net(self):
		dim = self.dim
		cnn_x = []
		cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
		for i in range(self.n_layer - 1):
			cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
			dim *= 2
		cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
		cnn_x = nn.Sequential(*cnn_x)
		return cnn_x

	def forward(self, x):
		outputs = []
		for model in self.cnns:
			outputs.append(model(x))
			x = self.downsample(x)
		return outputs

	def calc_dis_loss(self, input_fake, input_real):
		outs0 = self.forward(input_fake)
		outs1 = self.forward(input_real)
		loss = 0

		for it, (out0, out1) in enumerate(zip(outs0, outs1)):
			if self.gan_type == 'lsgan':
				loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)

			# elif self.gan_type == 'nsgan':
			# 	all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
			# 	all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
			# 	loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
			# 					   F.binary_cross_entropy(F.sigmoid(out1), all1))
			else:
				assert 0, "Unsupported GAN type: {}".format(self.gan_type)

		return loss

	def calc_gen_loss(self, input_fake):
		outs0 = self.forward(input_fake)

		loss = 0
		for it, (out0) in enumerate(outs0):
			if self.gan_type == 'lsgan':
				loss += torch.mean((out0 - 1)**2) # LSGAN

			# elif self.gan_type == 'nsgan':
			# 	all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
			# 	loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
			else:
				assert 0, "Unsupported GAN type: {}".format(self.gan_type)

		return loss
