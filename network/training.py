import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network.modules import Generator, Encoder, VGGDistance
from network.utils import ImageTensorDataset, AugmentedDataset

# stylegan2 modules
from model import Discriminator


class LatentModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.label_embedding = nn.Embedding(
			num_embeddings=config['n_labels'],
			embedding_dim=config['label_dim'],
			_weight=(2 * torch.rand(config['n_labels'], config['label_dim']) - 1) * 0.05
		)

		self.label_embedding = torch.nn.DataParallel(self.label_embedding)

		self.uncorrelated_embedding = nn.Embedding(
			num_embeddings=config['n_imgs'],
			embedding_dim=config['uncorrelated_dim'],
			_weight=(2 * torch.rand(config['n_imgs'], config['uncorrelated_dim']) - 1) * 0.05
		)

		self.uncorrelated_embedding = torch.nn.DataParallel(self.uncorrelated_embedding)

		latent_dim = config['label_dim'] + config['uncorrelated_dim']
		if config['correlation']:
			self.correlated_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['correlated_dim'])
			self.correlated_encoder = torch.nn.DataParallel(self.correlated_encoder)

			latent_dim += config['correlated_dim']

		self.generator = Generator(latent_dim=latent_dim, img_size=config['img_shape'][0])
		self.generator = torch.nn.DataParallel(self.generator)


class AmortizedModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.label_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['label_dim'])
		self.label_encoder = torch.nn.DataParallel(self.label_encoder)

		self.uncorrelated_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['uncorrelated_dim'])
		self.uncorrelated_encoder = torch.nn.DataParallel(self.uncorrelated_encoder)

		latent_dim = config['label_dim'] + config['uncorrelated_dim']
		if config['correlation']:
			self.correlated_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['correlated_dim'])
			self.correlated_encoder = torch.nn.DataParallel(self.correlated_encoder)

			latent_dim += config['correlated_dim']

		self.generator = Generator(latent_dim=latent_dim, img_size=config['img_shape'][0])
		self.generator = torch.nn.DataParallel(self.generator)

		self.discriminator = Discriminator(size=config['img_shape'][0])
		self.discriminator = torch.nn.DataParallel(self.discriminator)


class Model:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.latent_model = None
		self.amortized_model = None

		self.reconstruction_loss = VGGDistance(
			layer_ids=config['perceptual_loss']['layers'],
			normalize=config['perceptual_loss']['normalize']
		)

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(checkpoint_dir):
		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		model = Model(config)

		if os.path.exists(os.path.join(checkpoint_dir, 'latent.pth')):
			model.latent_model = LatentModel(config)
			model.latent_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'latent.pth')))

		if os.path.exists(os.path.join(checkpoint_dir, 'amortized.pth')):
			model.amortized_model = AmortizedModel(config)
			model.amortized_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'amortized.pth')))

		return model

	def save(self, checkpoint_dir):
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)

		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		if self.latent_model:
			torch.save(self.latent_model.state_dict(), os.path.join(checkpoint_dir, 'latent.pth'))

		if self.amortized_model:
			torch.save(self.amortized_model.state_dict(), os.path.join(checkpoint_dir, 'amortized.pth'))

	def train_latent_model(self, imgs, labels, masks, model_dir, tensorboard_dir):
		self.latent_model = LatentModel(self.config)

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			label=torch.from_numpy(labels.astype(np.int64))
		)

		if self.config['correlation']:
			if self.config['correlation'] == 'localized':
				data['mask'] = torch.from_numpy(masks).unsqueeze(dim=1)

			dataset = AugmentedDataset(data, augmentation=self.config['correlation_augmentation'])
		else:
			dataset = ImageTensorDataset(data)

		data_loader = DataLoader(
			dataset, batch_size=self.config['disentanglement']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=False
		)

		trainable_params = [
			{
				'params': itertools.chain(
					self.latent_model.label_embedding.parameters(),
					self.latent_model.uncorrelated_embedding.parameters()
				),

				'lr': self.config['disentanglement']['learning_rate']['latent']
			},
			{
				'params': self.latent_model.generator.parameters(),
				'lr': self.config['disentanglement']['learning_rate']['generator']
			}
		]

		if self.config['correlation']:
			trainable_params.append({
				'params': self.latent_model.correlated_encoder.parameters(),
				'lr': self.config['disentanglement']['learning_rate']['encoder']
			})

		optimizer = Adam(trainable_params, betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['disentanglement']['n_epochs'] * len(data_loader),
			eta_min=self.config['disentanglement']['learning_rate']['min']
		)

		self.latent_model.to(self.device)
		self.reconstruction_loss.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(1, self.config['disentanglement']['n_epochs'] + 1):
			self.latent_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses = self.__iterate_latent_model(batch)
				loss = 0
				for term, val in losses.items():
					loss += self.config['disentanglement']['loss_weights'][term] * val

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()

				pbar.set_description_str('[disentanglement] epoch #{}'.format(epoch))
				pbar.set_postfix(loss=loss.item())

			pbar.close()

			summary.add_scalar(tag='loss', scalar_value=loss.item(), global_step=epoch)

			for term, val in losses.items():
				summary.add_scalar(tag='loss/{}'.format(term), scalar_value=val.item(), global_step=epoch)

			if epoch % self.config['disentanglement']['n_epochs_between_visualizations'] == 0:
				figure_fixed = self.__visualize_translation(dataset, randomized=False)
				figure_random = self.__visualize_translation(dataset, randomized=True)

				summary.add_image(tag='translation-fixed', img_tensor=figure_fixed, global_step=epoch)
				summary.add_image(tag='translation-random', img_tensor=figure_random, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def warmup_amortized_model(self, imgs, labels, masks, model_dir, tensorboard_dir):
		self.amortized_model = AmortizedModel(self.config)
		self.amortized_model.generator.load_state_dict(self.latent_model.generator.state_dict())

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			label=torch.from_numpy(labels.astype(np.int64))
		)

		if self.config['correlation']:
			if self.config['correlation'] == 'localized':
				data['mask'] = torch.from_numpy(masks).unsqueeze(dim=1)

			dataset = AugmentedDataset(data, augmentation=self.config['correlation_augmentation'])
		else:
			dataset = ImageTensorDataset(data)

		data_loader = DataLoader(
			dataset, batch_size=self.config['amortization']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=True
		)

		trainable_params = [
			self.amortized_model.label_encoder.parameters(),
			self.amortized_model.uncorrelated_encoder.parameters()
		]

		if self.config['correlation']:
			trainable_params.append(self.amortized_model.correlated_encoder.parameters())

		optimizer = Adam(
			params=itertools.chain(*trainable_params),
			lr=self.config['amortization']['learning_rate']['max']
		)

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['amortization']['n_epochs'] * len(data_loader),
			eta_min=self.config['amortization']['learning_rate']['min']
		)

		self.latent_model.to(self.device)
		self.amortized_model.to(self.device)

		os.mkdir(tensorboard_dir)
		summary = SummaryWriter(log_dir=tensorboard_dir)

		for epoch in range(1, self.config['amortization']['n_epochs'] + 1):
			self.latent_model.train()
			self.amortized_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses = self.__iterate_encoders(batch)
				loss = 0
				for term, val in losses.items():
					loss += val

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()

				pbar.set_description_str('[amortization] epoch #{}'.format(epoch))
				pbar.set_postfix(loss=loss.item())

			pbar.close()

			for term, val in losses.items():
				summary.add_scalar(tag='loss/{}'.format(term), scalar_value=val.item(), global_step=epoch)

			if epoch % self.config['amortization']['n_epochs_between_visualizations'] == 0:
				figure_fixed = self.__visualize_translation(dataset, randomized=False, amortized=True)
				figure_random = self.__visualize_translation(dataset, randomized=True, amortized=True)

				summary.add_image(tag='translation-fixed', img_tensor=figure_fixed, global_step=epoch)
				summary.add_image(tag='translation-random', img_tensor=figure_random, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def tune_amortized_model(self, imgs, labels, masks, model_dir, tensorboard_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			label=torch.from_numpy(labels.astype(np.int64))
		)

		if self.config['correlation']:
			if self.config['correlation'] == 'localized':
				data['mask'] = torch.from_numpy(masks).unsqueeze(dim=1)

			dataset = AugmentedDataset(data, augmentation=self.config['correlation_augmentation'])
		else:
			dataset = ImageTensorDataset(data)

		data_loader = DataLoader(
			dataset, batch_size=self.config['synthesis']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=True
		)

		trainable_params = [
			self.amortized_model.label_encoder.parameters(),
			self.amortized_model.uncorrelated_encoder.parameters(),
			self.amortized_model.generator.parameters()
		]

		if self.config['correlation']:  # TODO: maybe freeze
			trainable_params.append(self.amortized_model.correlated_encoder.parameters())

		generator_optimizer = Adam(
			params=itertools.chain(*trainable_params),
			lr=self.config['synthesis']['learning_rate']['generator'],
			betas=(0.5, 0.999)
		)

		discriminator_optimizer = Adam(
			params=self.amortized_model.discriminator.parameters(),
			lr=self.config['synthesis']['learning_rate']['discriminator'],
			betas=(0.5, 0.999)
		)

		self.latent_model.to(self.device)
		self.amortized_model.to(self.device)
		self.reconstruction_loss.to(self.device)

		os.mkdir(tensorboard_dir)
		summary = SummaryWriter(log_dir=tensorboard_dir)

		for epoch in range(1, self.config['synthesis']['n_epochs'] + 1):
			self.latent_model.train()
			self.amortized_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses_discriminator = self.__iterate_discriminator(batch)
				loss_discriminator = (
					losses_discriminator['fake']
					+ losses_discriminator['real']
					+ losses_discriminator['gradient_penalty']
				)

				generator_optimizer.zero_grad()
				discriminator_optimizer.zero_grad()
				loss_discriminator.backward()
				discriminator_optimizer.step()

				losses_generator = self.__iterate_amortized_model(batch)
				loss_generator = 0
				for term, val in losses_generator.items():
					loss_generator += self.config['synthesis']['loss_weights'][term] * val

				generator_optimizer.zero_grad()
				discriminator_optimizer.zero_grad()
				loss_generator.backward()
				generator_optimizer.step()

				pbar.set_description_str('[synthesis] epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

			pbar.close()

			summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)
			summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)

			for term, val in losses_generator.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=val.item(), global_step=epoch)

			if epoch % self.config['synthesis']['n_epochs_between_visualizations'] == 0:
				figure_fixed = self.__visualize_translation(dataset, randomized=False, amortized=True)
				figure_random = self.__visualize_translation(dataset, randomized=True, amortized=True)

				summary.add_image(tag='translation-fixed', img_tensor=figure_fixed, global_step=epoch)
				summary.add_image(tag='translation-random', img_tensor=figure_random, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def __iterate_latent_model(self, batch):
		label_code = self.latent_model.label_embedding(batch['label'])

		uncorrelated_code = self.latent_model.uncorrelated_embedding(batch['img_id'])
		if self.config['uncorrelated_std'] != 0:
			noise = torch.zeros_like(uncorrelated_code)
			noise.normal_(mean=0, std=self.config['uncorrelated_std'])

			uncorrelated_code_regularized = uncorrelated_code + noise
		else:
			uncorrelated_code_regularized = uncorrelated_code

		if self.config['correlation']:
			correlated_code = self.latent_model.correlated_encoder(batch['img_augmented'])
			latent_code = torch.cat((label_code, correlated_code, uncorrelated_code_regularized), dim=1)
		else:
			latent_code = torch.cat((label_code, uncorrelated_code_regularized), dim=1)

		img_reconstructed = self.latent_model.generator(latent_code)
		loss_reconstruction = self.reconstruction_loss(img_reconstructed, batch['img'])

		loss_uncorrelated_decay = torch.mean(uncorrelated_code ** 2, dim=1).mean()

		return {
			'reconstruction': loss_reconstruction,
			'uncorrelated_decay': loss_uncorrelated_decay
		}

	def __iterate_encoders(self, batch):
		with torch.no_grad():
			label_code_target = self.latent_model.label_embedding(batch['label'])
			uncorrelated_code_target = self.latent_model.uncorrelated_embedding(batch['img_id'])

			if self.config['correlation']:
				correlated_code_target = self.latent_model.correlated_encoder(batch['img_augmented'])

		label_code = self.amortized_model.label_encoder(batch['img'])
		uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])

		loss_label = torch.mean((label_code - label_code_target) ** 2, dim=1).mean()
		loss_uncorrelated = torch.mean((uncorrelated_code - uncorrelated_code_target) ** 2, dim=1).mean()

		losses = {
			'label': loss_label,
			'uncorrelated': loss_uncorrelated
		}

		if self.config['correlation']:
			correlated_code = self.amortized_model.correlated_encoder(batch['img'])
			loss_correlated = torch.mean((correlated_code - correlated_code_target) ** 2, dim=1).mean()

			losses['correlated'] = loss_correlated

		return losses

	def __iterate_amortized_model(self, batch):
		with torch.no_grad():
			label_code_target = self.latent_model.label_embedding(batch['label'])
			uncorrelated_code_target = self.latent_model.uncorrelated_embedding(batch['img_id'])

			if self.config['correlation']:
				correlated_code_target = self.latent_model.correlated_encoder(batch['img_augmented'])

		label_code = self.amortized_model.label_encoder(batch['img'])
		uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])

		loss_label = torch.mean((label_code - label_code_target) ** 2, dim=1).mean()
		loss_uncorrelated = torch.mean((uncorrelated_code - uncorrelated_code_target) ** 2, dim=1).mean()

		if self.config['correlation']:
			correlated_code = self.amortized_model.correlated_encoder(batch['img'])
			loss_correlated = torch.mean((correlated_code - correlated_code_target) ** 2, dim=1).mean()

			latent_code = torch.cat((label_code, correlated_code, uncorrelated_code), dim=1)
		else:
			latent_code = torch.cat((label_code, uncorrelated_code), dim=1)

		img_reconstructed = self.amortized_model.generator(latent_code)
		loss_reconstruction = self.reconstruction_loss(img_reconstructed, batch['img'])

		discriminator_fake = self.amortized_model.discriminator(img_reconstructed)
		loss_adversarial = self.__adv_loss(discriminator_fake, 1)

		losses = {
			'reconstruction': loss_reconstruction,
			'latent': loss_label + loss_uncorrelated,
			'adversarial': loss_adversarial
		}

		if self.config['correlation']:
			losses['latent'] += loss_correlated

		return losses

	def __iterate_discriminator(self, batch):
		with torch.no_grad():
			label_code = self.amortized_model.label_encoder(batch['img'])
			uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])

			if self.config['correlation']:
				correlated_code = self.amortized_model.correlated_encoder(batch['img'])
				latent_code = torch.cat((label_code, correlated_code, uncorrelated_code), dim=1)
			else:
				latent_code = torch.cat((label_code, uncorrelated_code), dim=1)

			img_reconstructed = self.amortized_model.generator(latent_code)

		batch['img'].requires_grad_()  # for gradient penalty
		discriminator_fake = self.amortized_model.discriminator(img_reconstructed)
		discriminator_real = self.amortized_model.discriminator(batch['img'])

		loss_fake = self.__adv_loss(discriminator_fake, 0)
		loss_real = self.__adv_loss(discriminator_real, 1)
		loss_gp = self.__gradient_penalty(discriminator_real, batch['img'])

		return {
			'fake': loss_fake,
			'real': loss_real,
			'gradient_penalty': loss_gp
		}

	@staticmethod
	def __adv_loss(logits, target):
		targets = torch.full_like(logits, fill_value=target)
		loss = F.binary_cross_entropy_with_logits(logits, targets)
		return loss

	@staticmethod
	def __gradient_penalty(d_out, x_in):
		batch_size = x_in.size(0)

		grad_dout = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
		grad_dout2 = grad_dout.pow(2)

		reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
		return reg

	@torch.no_grad()
	def __visualize_translation(self, dataset, n_samples=10, randomized=False, amortized=False):
		random = self.rs if randomized else np.random.RandomState(seed=0)

		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
		batch = dataset[img_idx]
		batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

		if amortized:
			self.amortized_model.eval()

			label_code = self.amortized_model.label_encoder(batch['img'])
			uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])

			if self.config['correlation']:
				correlated_code = self.amortized_model.correlated_encoder(batch['img'])

		else:
			self.latent_model.eval()

			label_code = self.latent_model.label_embedding(batch['label'])
			uncorrelated_code = self.latent_model.uncorrelated_embedding(batch['img_id'])

			if self.config['correlation']:
				correlated_code = self.latent_model.correlated_encoder(batch['img_augmented'])

		generator = self.amortized_model.generator if amortized else self.latent_model.generator

		blank = torch.ones_like(batch['img'][0])
		summary = [torch.cat([blank] + list(batch['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [batch['img'][i]]

			for j in range(n_samples):
				if self.config['correlation']:
					latent_code = torch.cat([label_code[i], correlated_code[i], uncorrelated_code[j]], dim=0)
				else:
					latent_code = torch.cat([label_code[i], uncorrelated_code[j]], dim=0)

				converted_img = generator(latent_code.unsqueeze(dim=0))
				converted_imgs.append(converted_img[0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		return summary.clamp(min=0, max=1)
