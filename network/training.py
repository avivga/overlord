import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from network.modules import Generator, Encoder, VGGFeatures, VGGDistance, VGGStyle
from network.utils import NamedTensorDataset, AugmentedDataset

from model import Discriminator


class Model:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.content_embedding = nn.Embedding(num_embeddings=config['n_imgs'], embedding_dim=config['content_dim'])
		self.class_embedding = nn.Embedding(num_embeddings=config['n_classes'], embedding_dim=config['class_dim'])

		nn.init.uniform_(self.content_embedding.weight, a=-0.05, b=0.05)
		nn.init.uniform_(self.class_embedding.weight, a=-0.05, b=0.05)

		self.generator = Generator(
			img_size=config['img_shape'][0],
			content_dim=config['content_dim'],
			class_dim=config['class_dim'],
			style_dim=config['style_dim'],
			style_descriptor_dim=config['style_descriptor']['dim']
		)

		self.discriminator = Discriminator(size=config['img_shape'][0])
		self.content_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['content_dim'])
		self.class_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['class_dim'])

		self.vgg_features = VGGFeatures()
		self.perceptual_loss = VGGDistance(self.vgg_features, config['perceptual_loss']['layers'])
		self.style_descriptor = VGGStyle(self.vgg_features, config['style_descriptor']['layers'])

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(checkpoint_dir):
		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		model = Model(config)

		model.content_embedding.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'content_embedding.pth')))
		model.class_embedding.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'class_embedding.pth')))

		model.generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'generator.pth')))
		model.discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'discriminator.pth')))

		model.content_encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'content_encoder.pth')))
		model.class_encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'class_encoder.pth')))

		return model

	def save(self, checkpoint_dir):
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)

		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		torch.save(self.content_embedding.state_dict(), os.path.join(checkpoint_dir, 'content_embedding.pth'))
		torch.save(self.class_embedding.state_dict(), os.path.join(checkpoint_dir, 'class_embedding.pth'))

		torch.save(self.generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
		torch.save(self.discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))

		torch.save(self.content_encoder.state_dict(), os.path.join(checkpoint_dir, 'content_encoder.pth'))
		torch.save(self.class_encoder.state_dict(), os.path.join(checkpoint_dir, 'class_encoder.pth'))

	def train(self, imgs, classes, model_dir, tensorboard_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = AugmentedDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=False
		)

		optimizer = Adam([
			{
				'params': itertools.chain(
					self.content_embedding.parameters(),
					self.class_embedding.parameters()
				),

				'lr': self.config['train']['learning_rate']['latent']
			},
			{
				'params': self.generator.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		self.generator.to(self.device)
		self.vgg_features.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['train']['n_epochs']):
			self.generator.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch['content_code'] = self.content_embedding(batch['img_id'])
				batch['class_code'] = self.class_embedding(batch['class_id'])
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses_generator = self.train_latent_generator(batch)
				loss_generator = 0
				for term, loss in losses_generator.items():
					loss_generator += self.config['train']['loss_weights'][term] * loss

				optimizer.zero_grad()
				loss_generator.backward()
				optimizer.step()
				scheduler.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item())

			pbar.close()

			summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)

			for term, loss in losses_generator.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			samples_fixed = self.generate_samples(dataset, randomized=False)
			samples_random = self.generate_samples(dataset, randomized=True)

			summary.add_image(tag='samples-fixed', img_tensor=samples_fixed, global_step=epoch)
			summary.add_image(tag='samples-random', img_tensor=samples_random, global_step=epoch)

			if epoch % 10 == 0:
				content_codes = self.extract_codes(dataset)
				score_train, score_test = self.classification_score(X=content_codes, y=classes)
				summary.add_scalar(tag='class_from_content/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_content/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def amortize(self, imgs, classes, model_dir, tensorboard_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['amortize']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=True
		)

		generator_optimizer = Adam(
			params=itertools.chain(
				self.content_encoder.parameters(),
				self.class_encoder.parameters(),
				self.generator.parameters()
			),

			lr=self.config['amortize']['learning_rate']['generator'],
			betas=(0.5, 0.999)
		)

		discriminator_optimizer = Adam(
			params=self.discriminator.parameters(),
			lr=self.config['amortize']['learning_rate']['discriminator'],
			betas=(0.5, 0.999)
		)

		self.generator.to(self.device)
		self.discriminator.to(self.device)
		self.content_encoder.to(self.device)
		self.class_encoder.to(self.device)
		self.vgg_features.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['amortize']['n_epochs']):
			self.generator.train()
			self.discriminator.train()
			self.content_encoder.train()
			self.class_encoder.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch['content_code'] = self.content_embedding(batch['img_id'])
				batch['class_code'] = self.class_embedding(batch['class_id'])
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses_discriminator = self.train_discriminator(batch)
				loss_discriminator = (
					losses_discriminator['fake']
					+ losses_discriminator['real']
					+ losses_discriminator['gradient_penalty']
				)

				generator_optimizer.zero_grad()
				discriminator_optimizer.zero_grad()
				loss_discriminator.backward()
				discriminator_optimizer.step()

				losses_generator = self.train_amortized_generator(batch)
				loss_generator = 0
				for term, loss in losses_generator.items():
					loss_generator += self.config['amortize']['loss_weights'][term] * loss

				generator_optimizer.zero_grad()
				discriminator_optimizer.zero_grad()
				loss_generator.backward()
				generator_optimizer.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

			pbar.close()

			summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)
			summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)

			for term, loss in losses_generator.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			samples_fixed = self.generate_samples(dataset, randomized=False, amortized=True)
			samples_random = self.generate_samples(dataset, randomized=True, amortized=True)

			summary.add_image(tag='samples-fixed', img_tensor=samples_fixed, global_step=epoch)
			summary.add_image(tag='samples-random', img_tensor=samples_random, global_step=epoch)

			if epoch % 10 == 0:
				content_codes = self.extract_codes(dataset, amortized=True)
				score_train, score_test = self.classification_score(X=content_codes, y=classes)
				summary.add_scalar(tag='class_from_content/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_content/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir)

		summary.close()

	@torch.no_grad()
	def translate(self, imgs, classes, n_translations_per_image, out_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		unique_classes = np.unique(classes)

		class_img_ids = {
			class_id: np.where(classes == class_id)[0]
			for class_id in unique_classes
		}

		self.content_encoder.eval()
		self.class_encoder.eval()
		self.generator.eval()

		self.content_encoder.to(self.device)
		self.class_encoder.to(self.device)
		self.generator.to(self.device)
		self.vgg_features.to(self.device)

		rs = np.random.RandomState(seed=1337)
		dataset = AugmentedDataset(data)
		for source_class, target_class in itertools.product(unique_classes, unique_classes):
			if source_class == target_class:
				continue

			translation_dir = os.path.join(out_dir, '{}-to-{}'.format(source_class, target_class))
			os.mkdir(translation_dir)

			os.mkdir(os.path.join(translation_dir, 'content'))
			os.mkdir(os.path.join(translation_dir, 'style'))
			os.mkdir(os.path.join(translation_dir, 'translation'))

			pbar = tqdm(class_img_ids[source_class])
			pbar.set_description_str('translating {} to {}'.format(source_class, target_class))

			for content_idx in pbar:
				style_idxs = rs.choice(class_img_ids[target_class], size=n_translations_per_image, replace=False)

				content_imgs = torch.stack([dataset[content_idx]['img']] * n_translations_per_image, dim=0)
				style_imgs = dataset[style_idxs]['img']

				content_codes = self.content_encoder(content_imgs.to(self.device))
				class_codes = self.class_encoder(style_imgs.to(self.device))
				style_descriptors = self.style_descriptor(style_imgs.to(self.device))

				translated_imgs = self.generator(content_codes, class_codes, style_descriptors).cpu()
				for i in range(n_translations_per_image):
					torchvision.utils.save_image(content_imgs[i],
						os.path.join(translation_dir, 'content', '{}.png'.format(content_idx))
					)

					torchvision.utils.save_image(
						style_imgs[i],
						os.path.join(translation_dir, 'style', '{}.png'.format(style_idxs[i]))
					)

					torchvision.utils.save_image(
						translated_imgs[i],
						os.path.join(translation_dir, 'translation', '{}-{}.png'.format(content_idx, style_idxs[i]))
					)

	def train_latent_generator(self, batch):
		with torch.no_grad():
			style_descriptor = self.style_descriptor(batch['img_augmented'])

		if self.config['content_std'] != 0:
			noise = torch.zeros_like(batch['content_code'])
			noise.normal_(mean=0, std=self.config['content_std'])

			content_code_regularized = batch['content_code'] + noise
		else:
			content_code_regularized = batch['content_code']

		img_reconstructed = self.generator(content_code_regularized, batch['class_code'], style_descriptor)
		loss_reconstruction = self.perceptual_loss(img_reconstructed, batch['img'])

		loss_content_decay = torch.sum(batch['content_code'] ** 2, dim=1).mean()

		return {
			'reconstruction': loss_reconstruction,
			'content_decay': loss_content_decay
		}

	def train_amortized_generator(self, batch):
		with torch.no_grad():
			style_descriptor = self.style_descriptor(batch['img'])

		content_code = self.content_encoder(batch['img'])
		class_code = self.class_encoder(batch['img'])
		img_reconstructed = self.generator(content_code, class_code, style_descriptor)
		loss_reconstruction = self.perceptual_loss(img_reconstructed, batch['img'])

		loss_content = torch.mean((content_code - batch['content_code']) ** 2, dim=1).mean()
		loss_class = torch.mean((class_code - batch['class_code']) ** 2, dim=1).mean()

		discriminator_fake = self.discriminator(img_reconstructed)
		loss_adversarial = self.adv_loss(discriminator_fake, 1)

		return {
			'reconstruction': loss_reconstruction,
			'latent': loss_content + loss_class,
			'adversarial': loss_adversarial
		}

	def train_discriminator(self, batch):
		with torch.no_grad():
			content_code = self.content_encoder(batch['img'])
			class_code = self.class_encoder(batch['img'])
			style_descriptor = self.style_descriptor(batch['img'])
			img_reconstructed = self.generator(content_code, class_code, style_descriptor)

		batch['img'].requires_grad_()  # for gradient penalty
		discriminator_fake = self.discriminator(img_reconstructed)
		discriminator_real = self.discriminator(batch['img'])

		loss_fake = self.adv_loss(discriminator_fake, 0)
		loss_real = self.adv_loss(discriminator_real, 1)
		loss_gp = self.gradient_penalty(discriminator_real, batch['img'])

		return {
			'fake': loss_fake,
			'real': loss_real,
			'gradient_penalty': loss_gp
		}

	def adv_loss(self, logits, target):
		assert target in [1, 0]
		targets = torch.full_like(logits, fill_value=target)
		loss = F.binary_cross_entropy_with_logits(logits, targets)
		return loss

	def gradient_penalty(self, d_out, x_in):
		batch_size = x_in.size(0)
		grad_dout = torch.autograd.grad(
			outputs=d_out.sum(), inputs=x_in,
			create_graph=True, retain_graph=True, only_inputs=True
		)[0]
		grad_dout2 = grad_dout.pow(2)
		assert(grad_dout2.size() == x_in.size())
		reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
		return reg

	@torch.no_grad()
	def generate_samples(self, dataset, n_samples=10, randomized=False, amortized=False):
		self.content_encoder.eval()
		self.class_encoder.eval()
		self.generator.eval()

		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
		samples = dataset[img_idx]

		if amortized:
			samples['content_code'] = self.content_encoder(samples['img'].to(self.device))
			samples['class_code'] = self.class_encoder(samples['img'].to(self.device))

		else:
			samples['content_code'] = self.content_embedding(samples['img_id'])
			samples['class_code'] = self.class_embedding(samples['class_id'])

		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
		samples['style_descriptor'] = self.style_descriptor(samples['img'])

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				converted_img = self.generator(
					samples['content_code'][[j]],
					samples['class_code'][[i]],
					samples['style_descriptor'][[i]]
				)

				converted_imgs.append(converted_img[0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		return summary.clamp(min=0, max=1)

	@torch.no_grad()
	def extract_codes(self, dataset, amortized=False):
		data_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)

		content_codes = []
		for batch in data_loader:
			if amortized:
				batch_content_codes = self.content_encoder(batch['img'].to(self.device))
			else:
				batch_content_codes = self.content_embedding(batch['img_id'])

			content_codes.append(batch_content_codes.cpu().numpy())

		content_codes = np.concatenate(content_codes, axis=0)
		return content_codes

	@staticmethod
	def classification_score(X, y):
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

		classifier = LogisticRegression(random_state=0)
		classifier.fit(X_train, y_train)

		acc_train = classifier.score(X_train, y_train)
		acc_test = classifier.score(X_test, y_test)

		return acc_train, acc_test
