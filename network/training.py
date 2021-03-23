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

from network.modules import Generator, Encoder, VGGFeatures, VGGDistance
from network.utils import NamedTensorDataset, AugmentedDataset

from model import Discriminator


class LatentModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.uncorrelated_embedding = nn.Embedding(num_embeddings=config['n_imgs'], embedding_dim=config['uncorrelated_dim'])
		self.class_embedding = nn.Embedding(num_embeddings=config['n_classes'], embedding_dim=config['class_dim'])

		nn.init.uniform_(self.uncorrelated_embedding.weight, a=-0.05, b=0.05)
		nn.init.uniform_(self.class_embedding.weight, a=-0.05, b=0.05)

		self.correlated_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['correlated_dim'])

		self.generator = Generator(
			img_size=config['img_shape'][0],
			uncorrelated_dim=config['uncorrelated_dim'],
			class_dim=config['class_dim'],
			correlated_dim=config['correlated_dim']
		)


class AmortizedModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.uncorrelated_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['uncorrelated_dim'])
		self.class_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['class_dim'])
		self.correlated_encoder = Encoder(img_size=config['img_shape'][0], code_dim=config['correlated_dim'])

		self.generator = Generator(
			img_size=config['img_shape'][0],
			uncorrelated_dim=config['uncorrelated_dim'],
			class_dim=config['class_dim'],
			correlated_dim=config['correlated_dim']
		)

		self.discriminator = Discriminator(size=config['img_shape'][0])


class Model:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.latent_model = None
		self.amortized_model = None
		self.amortized_model_ema = None

		self.vgg_features = VGGFeatures()
		self.perceptual_loss = VGGDistance(self.vgg_features, config['perceptual_loss']['layers'])

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

		if os.path.exists(os.path.join(checkpoint_dir, 'amortized_ema.pth')):
			model.amortized_model_ema = AmortizedModel(config)
			model.amortized_model_ema.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'amortized_ema.pth')))

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

		if self.amortized_model_ema:
			torch.save(self.amortized_model_ema.state_dict(), os.path.join(checkpoint_dir, 'amortized_ema.pth'))

	def train(self, imgs, classes, model_dir, tensorboard_dir):
		self.latent_model = LatentModel(self.config)

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
					self.latent_model.uncorrelated_embedding.parameters(),
					self.latent_model.class_embedding.parameters()
				),

				'lr': self.config['train']['learning_rate']['latent']
			},
			{
				'params': self.latent_model.correlated_encoder.parameters(),
				'lr': self.config['train']['learning_rate']['encoder']
			},
			{
				'params': self.latent_model.generator.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		self.latent_model.correlated_encoder.to(self.device)
		self.latent_model.generator.to(self.device)
		self.vgg_features.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['train']['n_epochs']):
			self.latent_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch['uncorrelated_code'] = self.latent_model.uncorrelated_embedding(batch['img_id'])
				batch['class_code'] = self.latent_model.class_embedding(batch['class_id'])
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses = self.train_latent_generator(batch)
				loss_total = 0
				for term, loss in losses.items():
					loss_total += self.config['train']['loss_weights'][term] * loss

				optimizer.zero_grad()
				loss_total.backward()
				optimizer.step()
				scheduler.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=loss_total.item())

			pbar.close()

			summary.add_scalar(tag='loss/generator', scalar_value=loss_total.item(), global_step=epoch)

			for term, loss in losses.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			fig_translation_fixed = self.visualize_translation(dataset, randomized=False)
			fig_translation_random = self.visualize_translation(dataset, randomized=True)

			summary.add_image(tag='translation-fixed', img_tensor=fig_translation_fixed, global_step=epoch)
			summary.add_image(tag='translation-random', img_tensor=fig_translation_random, global_step=epoch)

			if epoch % 10 == 0:
				uncorrelated_codes = self.encode_uncorrelated(dataset)
				score_train, score_test = self.classification_score(X=uncorrelated_codes, y=classes)
				summary.add_scalar(tag='class_from_uncorrelated/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_uncorrelated/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def amortize(self, imgs, classes, model_dir, tensorboard_dir):
		self.amortized_model = AmortizedModel(self.config)
		self.warmup(imgs, classes, model_dir, tensorboard_dir)

		self.amortized_model_ema = AmortizedModel(self.config)
		self.amortized_model_ema.load_state_dict(self.amortized_model.state_dict())

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
				self.amortized_model.uncorrelated_encoder.parameters(),
				self.amortized_model.class_encoder.parameters(),
				self.amortized_model.generator.parameters()
			),

			lr=self.config['amortize']['learning_rate']['generator'],
			betas=(0.5, 0.999)
		)

		discriminator_optimizer = Adam(
			params=self.amortized_model.discriminator.parameters(),
			lr=self.config['amortize']['learning_rate']['discriminator'],
			betas=(0.5, 0.999)
		)

		self.amortized_model.to(self.device)
		self.amortized_model_ema.to(self.device)
		self.vgg_features.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['amortize']['n_epochs']):
			self.amortized_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch['uncorrelated_code'] = self.latent_model.uncorrelated_embedding(batch['img_id'])
				batch['class_code'] = self.latent_model.class_embedding(batch['class_id'])
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

				for param, param_ema in zip(self.amortized_model.parameters(), self.amortized_model_ema.parameters()):
					param_ema.data = torch.lerp(param.data, param_ema.data, weight=0.999)

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

			pbar.close()

			summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)
			summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)

			for term, loss in losses_generator.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			fig_translation_fixed = self.visualize_translation(dataset, randomized=False, amortized=True)
			fig_translation_random = self.visualize_translation(dataset, randomized=True, amortized=True)

			summary.add_image(tag='translation-fixed', img_tensor=fig_translation_fixed, global_step=epoch)
			summary.add_image(tag='translation-random', img_tensor=fig_translation_random, global_step=epoch)

			if epoch % 10 == 0:
				uncorrelated_codes = self.encode_uncorrelated(dataset, amortized=True)
				score_train, score_test = self.classification_score(X=uncorrelated_codes, y=classes)
				summary.add_scalar(tag='class_from_uncorrelated/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_uncorrelated/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def warmup(self, imgs, classes, model_dir, tensorboard_dir):
		self.amortized_model.correlated_encoder.load_state_dict(self.latent_model.correlated_encoder.state_dict())
		self.amortized_model.generator.load_state_dict(self.latent_model.generator.state_dict())

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['warmup']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=True
		)

		optimizer = Adam(
			params=itertools.chain(
				self.amortized_model.uncorrelated_encoder.parameters(),
				self.amortized_model.class_encoder.parameters(),
			),

			lr=self.config['warmup']['learning_rate']['encoder'],
			betas=(0.5, 0.999)
		)

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['warmup']['n_epochs'] * len(data_loader),
			eta_min=self.config['warmup']['learning_rate']['min']
		)

		self.amortized_model.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['warmup']['n_epochs']):
			self.amortized_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch['uncorrelated_code'] = self.latent_model.uncorrelated_embedding(batch['img_id'])
				batch['class_code'] = self.latent_model.class_embedding(batch['class_id'])
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses_encoders = self.train_encoders(batch)
				loss_encoders = 0
				for term, loss in losses_encoders.items():
					loss_encoders += self.config['warmup']['loss_weights'][term] * loss

				optimizer.zero_grad()
				loss_encoders.backward()
				optimizer.step()
				scheduler.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_encoders.item())

			pbar.close()

			for term, loss in losses_encoders.items():
				summary.add_scalar(tag='loss/warmup/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			fig_translation_fixed = self.visualize_translation(dataset, randomized=False, amortized=True)
			fig_translation_random = self.visualize_translation(dataset, randomized=True, amortized=True)

			summary.add_image(tag='warmup-translation-fixed', img_tensor=fig_translation_fixed, global_step=epoch)
			summary.add_image(tag='warmup-translation-random', img_tensor=fig_translation_random, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def train_latent_generator(self, batch):
		if self.config['uncorrelated_std'] != 0:
			noise = torch.zeros_like(batch['uncorrelated_code'])
			noise.normal_(mean=0, std=self.config['uncorrelated_std'])

			uncorrelated_code_regularized = batch['uncorrelated_code'] + noise
		else:
			uncorrelated_code_regularized = batch['uncorrelated_code']

		correlated_code = self.latent_model.correlated_encoder(batch['img_augmented'])

		img_reconstructed = self.latent_model.generator(uncorrelated_code_regularized, batch['class_code'], correlated_code)
		loss_reconstruction = self.perceptual_loss(img_reconstructed, batch['img'])

		loss_uncorrelated_decay = torch.mean(batch['uncorrelated_code'] ** 2, dim=1).mean()

		return {
			'reconstruction': loss_reconstruction,
			'uncorrelated_decay': loss_uncorrelated_decay
		}

	def train_encoders(self, batch):
		uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])
		class_code = self.amortized_model.class_encoder(batch['img'])

		loss_uncorrelated = torch.mean((uncorrelated_code - batch['uncorrelated_code']) ** 2, dim=1).mean()
		loss_class = torch.mean((class_code - batch['class_code']) ** 2, dim=1).mean()

		return {
			'latent': loss_uncorrelated + loss_class
		}

	def train_amortized_generator(self, batch):
		uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])
		class_code = self.amortized_model.class_encoder(batch['img'])

		with torch.no_grad():
			correlated_code = self.amortized_model.correlated_encoder(batch['img'])

		img_reconstructed = self.amortized_model.generator(uncorrelated_code, class_code, correlated_code)
		loss_reconstruction = self.perceptual_loss(img_reconstructed, batch['img'])

		loss_uncorrelated = torch.mean((uncorrelated_code - batch['uncorrelated_code']) ** 2, dim=1).mean()
		loss_class = torch.mean((class_code - batch['class_code']) ** 2, dim=1).mean()

		discriminator_fake = self.amortized_model.discriminator(img_reconstructed)
		loss_adversarial = self.adv_loss(discriminator_fake, 1)

		return {
			'reconstruction': loss_reconstruction,
			'latent': loss_uncorrelated + loss_class,
			'adversarial': loss_adversarial
		}

	def train_discriminator(self, batch):
		with torch.no_grad():
			uncorrelated_code = self.amortized_model.uncorrelated_encoder(batch['img'])
			class_code = self.amortized_model.class_encoder(batch['img'])
			correlated_code = self.amortized_model.correlated_encoder(batch['img'])
			img_reconstructed = self.amortized_model.generator(uncorrelated_code, class_code, correlated_code)

		batch['img'].requires_grad_()  # for gradient penalty
		discriminator_fake = self.amortized_model.discriminator(img_reconstructed)
		discriminator_real = self.amortized_model.discriminator(batch['img'])

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
	def translate_full(self, imgs, classes, n_translations_per_image, out_dir):
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

		self.amortized_model.to(self.device)
		self.amortized_model.eval()

		rs = np.random.RandomState(seed=1337)
		dataset = NamedTensorDataset(data)
		for source_class, target_class in itertools.product(unique_classes, unique_classes):
			if source_class == target_class:
				continue

			translation_dir = os.path.join(out_dir, '{}-to-{}'.format(source_class, target_class))
			os.mkdir(translation_dir)

			os.mkdir(os.path.join(translation_dir, 'uncorrelated'))
			os.mkdir(os.path.join(translation_dir, 'correlated'))
			os.mkdir(os.path.join(translation_dir, 'translation'))

			pbar = tqdm(class_img_ids[source_class])
			pbar.set_description_str('translating {} to {}'.format(source_class, target_class))

			for uncorrelated_idx in pbar:
				correlated_idxs = rs.choice(class_img_ids[target_class], size=n_translations_per_image, replace=False)

				uncorrelated_imgs = torch.stack([dataset[uncorrelated_idx]['img']] * n_translations_per_image, dim=0)
				correlated_imgs = dataset[correlated_idxs]['img']

				uncorrelated_codes = self.amortized_model.uncorrelated_encoder(uncorrelated_imgs.to(self.device))
				class_codes = self.amortized_model.class_encoder(correlated_imgs.to(self.device))
				correlated_codes = self.amortized_model.correlated_encoder(correlated_imgs.to(self.device))

				translated_imgs = self.amortized_model.generator(uncorrelated_codes, class_codes, correlated_codes).cpu()
				for i in range(n_translations_per_image):
					torchvision.utils.save_image(
						uncorrelated_imgs[i],
						os.path.join(translation_dir, 'uncorrelated', '{}.png'.format(uncorrelated_idx))
					)

					torchvision.utils.save_image(
						correlated_imgs[i],
						os.path.join(translation_dir, 'correlated', '{}.png'.format(correlated_idxs[i]))
					)

					torchvision.utils.save_image(
						translated_imgs[i],
						os.path.join(translation_dir, 'translation', '{}-{}.png'.format(uncorrelated_idx, correlated_idxs[i]))
					)

	@torch.no_grad()
	def translate(self, imgs, classes, n_translations_per_image, out_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		self.amortized_model.to(self.device)
		self.amortized_model.eval()

		rs = np.random.RandomState(seed=1337)
		dataset = NamedTensorDataset(data)

		os.mkdir(os.path.join(out_dir, 'uncorrelated'))
		os.mkdir(os.path.join(out_dir, 'correlated'))
		os.mkdir(os.path.join(out_dir, 'translation'))

		all_idx = np.arange(data['img'].shape[0])
		for uncorrelated_idx in tqdm(all_idx):
			correlated_idxs = rs.choice(np.delete(all_idx, uncorrelated_idx), size=n_translations_per_image, replace=False)

			uncorrelated_imgs = torch.stack([dataset[uncorrelated_idx]['img']] * n_translations_per_image, dim=0)
			correlated_imgs = dataset[correlated_idxs]['img']

			uncorrelated_codes = self.amortized_model.uncorrelated_encoder(uncorrelated_imgs.to(self.device))
			class_codes = self.amortized_model.class_encoder(correlated_imgs.to(self.device))
			correlated_codes = self.amortized_model.correlated_encoder(correlated_imgs.to(self.device))

			translated_imgs = self.amortized_model.generator(uncorrelated_codes, class_codes, correlated_codes).cpu()
			for i in range(n_translations_per_image):
				torchvision.utils.save_image(
					uncorrelated_imgs[i],
					os.path.join(out_dir, 'uncorrelated', '{}.png'.format(uncorrelated_idx)))

				torchvision.utils.save_image(
					correlated_imgs[i],
					os.path.join(out_dir, 'correlated', '{}.png'.format(correlated_idxs[i]))
				)

				torchvision.utils.save_image(
					translated_imgs[i],
					os.path.join(out_dir, 'translation', '{}-{}.png'.format(uncorrelated_idx, correlated_idxs[i]))
				)

	@torch.no_grad()
	def summary(self, imgs, classes, n_summaries, summary_size, out_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		self.amortized_model.to(self.device)

		for i in range(n_summaries):
			summary_img = self.visualize_translation(dataset, summary_size, randomized=True, amortized=True)
			torchvision.utils.save_image(summary_img, os.path.join(out_dir, '{}.png'.format(i)))

	@torch.no_grad()
	def encode(self, imgs, classes, amortized, out_path):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)

		if amortized:
			self.amortized_model.to(self.device)
		else:
			self.latent_model.correlated_encoder.to(self.device)
			self.latent_model.generator.to(self.device)

		np.savez(
			file=out_path,
			uncorrelated_codes=self.encode_uncorrelated(dataset, amortized),
			class_codes=self.encode_class(dataset, amortized),
			correlated_codes=self.encode_correlated(dataset, amortized),
			class_ids=classes
		)

	@torch.no_grad()
	def visualize_translation(self, dataset, n_samples=10, randomized=False, amortized=False):
		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
		samples = dataset[img_idx]

		if amortized:
			self.amortized_model.eval()

			samples['uncorrelated_code'] = self.amortized_model.uncorrelated_encoder(samples['img'].to(self.device))
			samples['class_code'] = self.amortized_model.class_encoder(samples['img'].to(self.device))
			samples['correlated_code'] = self.amortized_model.correlated_encoder(samples['img'].to(self.device))

		else:
			self.latent_model.eval()

			samples['uncorrelated_code'] = self.latent_model.uncorrelated_embedding(samples['img_id'])
			samples['class_code'] = self.latent_model.class_embedding(samples['class_id'])
			samples['correlated_code'] = self.latent_model.correlated_encoder(samples['img'].to(self.device))

		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				generator = self.amortized_model.generator if amortized else self.latent_model.generator
				converted_img = generator(samples['uncorrelated_code'][[j]], samples['class_code'][[i]], samples['correlated_code'][[i]])
				converted_imgs.append(converted_img[0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		return summary.clamp(min=0, max=1)

	@torch.no_grad()
	def encode_uncorrelated(self, dataset, amortized=False):
		if amortized:
			self.amortized_model.eval()
		else:
			self.latent_model.eval()

		uncorrelated_codes = []
		data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
		for batch in data_loader:
			if amortized:
				batch_uncorrelated_codes = self.amortized_model.uncorrelated_encoder(batch['img'].to(self.device))
			else:
				batch_uncorrelated_codes = self.latent_model.uncorrelated_embedding(batch['img_id'])

			uncorrelated_codes.append(batch_uncorrelated_codes.cpu().numpy())

		uncorrelated_codes = np.concatenate(uncorrelated_codes, axis=0)
		return uncorrelated_codes

	@torch.no_grad()
	def encode_class(self, dataset, amortized=False):
		if amortized:
			self.amortized_model.eval()
		else:
			self.latent_model.eval()

		class_codes = []
		data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
		for batch in data_loader:
			if amortized:
				batch_class_codes = self.amortized_model.class_encoder(batch['img'].to(self.device))
			else:
				batch_class_codes = self.latent_model.class_embedding(batch['class_id'])

			class_codes.append(batch_class_codes.cpu().numpy())

		class_codes = np.concatenate(class_codes, axis=0)
		return class_codes

	@torch.no_grad()
	def encode_correlated(self, dataset, amortized=False):
		correlated_encoder = self.amortized_model.correlated_encoder if amortized else self.latent_model.correlated_encoder
		correlated_encoder.eval()

		correlated_codes = []
		data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
		for batch in data_loader:
			batch_correlated_codes = correlated_encoder(batch['img'].to(self.device))
			correlated_codes.append(batch_correlated_codes.cpu().numpy())

		correlated_codes = np.concatenate(correlated_codes, axis=0)
		return correlated_codes

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
