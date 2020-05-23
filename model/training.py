import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.nn import L1Loss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import Generator, Discriminator, StyleEncoder
from model.utils import NamedTensorDataset


class SLord:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.generator = Generator(self.config)
		self.generator.to(self.device)

		self.discriminator = Discriminator(self.config)
		self.discriminator.to(self.device)

		self.style_encoder = StyleEncoder(self.config)
		self.style_encoder.to(self.device)

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		slord = SLord(config)
		slord.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
		slord.discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth')))
		slord.style_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'style_encoder.pth')))

		return slord

	def save(self, model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		torch.save(self.generator.state_dict(), os.path.join(model_dir, 'generator.pth'))
		torch.save(self.discriminator.state_dict(), os.path.join(model_dir, 'discriminator.pth'))
		torch.save(self.style_encoder.state_dict(), os.path.join(model_dir, 'style_encoder.pth'))

	def train_latent(self, imgs, classes, contents, model_dir, tensorboard_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		class_img_ids = {
			class_id: np.where(classes == class_id)[0]
			for class_id in np.unique(classes)
		}

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=False
		)

		reconstruction_loss_fn = L1Loss()

		generator_optimizer = Adam([
			{
				'params': itertools.chain(
					self.generator.class_style_modulation.parameters(),
					self.generator.modulation.parameters(),
					self.generator.decoder.parameters(),
					self.style_encoder.parameters()
				),

				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(
					self.generator.content_embedding.parameters(),
					# self.generator.style_embedding.parameters(),
					self.generator.class_embedding.parameters()
				),

				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		discriminator_optimizer = Adam([
			{
				'params': self.discriminator.parameters(),
				'lr': self.config['train']['learning_rate']['discriminator']
			}
		], betas=(0.5, 0.999))

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['train']['n_epochs']):
			self.generator.train()
			self.discriminator.train()
			self.style_encoder.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				target_class_ids = np.random.choice(self.config['n_classes'], size=batch['img_id'].shape[0])
				batch['target_class_id'] = torch.from_numpy(target_class_ids).to(self.device)

				target_class_img_ids = np.array([np.random.choice(class_img_ids[class_id]) for class_id in target_class_ids])
				batch['target_class_img_id'] = torch.from_numpy(target_class_img_ids).to(self.device)
				batch['target_class_img'] = data['img'][target_class_img_ids].to(self.device)

				with torch.no_grad():
					style_code = self.style_encoder(batch['target_class_img'], batch['target_class_id'])
					out = self.generator(batch['img_id'], style_code, batch['target_class_id'])

				# batch['img'].requires_grad_()  # for gradient penalty
				discriminator_fake = self.discriminator(out['img'], batch['target_class_id'])
				discriminator_real = self.discriminator(batch['img'], batch['class_id'])

				loss_fake = self.adv_loss(discriminator_fake, 0)
				loss_real = self.adv_loss(discriminator_real, 1)
				# loss_gp = self.gradient_penalty(discriminator_real, batch['img'])

				loss_discriminator = loss_fake + loss_real  #+ self.config['train']['loss_weights']['gradient_penalty'] * loss_gp

				discriminator_optimizer.zero_grad()
				loss_discriminator.backward()
				discriminator_optimizer.step()

				style_code = self.style_encoder(batch['target_class_img'], batch['target_class_id'])
				out = self.generator(batch['img_id'], style_code, batch['target_class_id'])

				discriminator_fake = self.discriminator(out['img'], batch['target_class_id'])
				loss_adversarial = self.adv_loss(discriminator_fake, 1)

				style_code_reconstructed = self.style_encoder(out['img'], batch['target_class_id'])
				loss_style_reconstruction = reconstruction_loss_fn(style_code_reconstructed, style_code)

				style_code_original = self.style_encoder(batch['img'], batch['class_id'])
				out = self.generator(batch['img_id'], style_code_original, batch['class_id'])
				loss_reconstruction = reconstruction_loss_fn(out['img'], batch['img'])
				loss_content_decay = torch.sum(out['content_code'] ** 2, dim=1).mean()
				loss_style_decay = torch.sum(out['style_code'] ** 2, dim=1).mean()

				loss_generator = (
					self.config['train']['loss_weights']['reconstruction'] * loss_reconstruction
					+ self.config['train']['loss_weights']['content_decay'] * loss_content_decay
					+ self.config['train']['loss_weights']['style_decay'] * loss_style_decay
					+ self.config['train']['loss_weights']['adversarial'] * loss_adversarial
					+ self.config['train']['loss_weights']['style_reconstruction'] * loss_style_reconstruction
				)

				generator_optimizer.zero_grad()
				loss_generator.backward()
				generator_optimizer.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

			pbar.close()

			summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)
			summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)
			summary.add_scalar(tag='loss/reconstruction', scalar_value=loss_reconstruction.item(), global_step=epoch)
			summary.add_scalar(tag='loss/adversarial', scalar_value=loss_adversarial.item(), global_step=epoch)
			summary.add_scalar(tag='loss/style', scalar_value=loss_style_reconstruction.item(), global_step=epoch)

			samples_fixed = self.generate_samples(dataset, randomized=False)
			samples_random = self.generate_samples(dataset, randomized=True)

			summary.add_image(tag='samples-fixed', img_tensor=samples_fixed, global_step=epoch)
			summary.add_image(tag='samples-random', img_tensor=samples_random, global_step=epoch)

			styles_random = self.generate_samples(dataset, randomized=True, style_only=True)
			summary.add_image(tag='styles-random', img_tensor=styles_random, global_step=epoch)

			if epoch % 5 == 0:
				content_codes, style_codes = self.extract_codes(dataset)

				if contents is not None:
					if contents.dtype == np.int64:
						score_train, score_test = self.classification_score(X=content_codes, y=contents)
					else:
						score_train, score_test = self.regression_score(X=content_codes, y=contents)

					summary.add_scalar(tag='content_from_content/train', scalar_value=score_train, global_step=epoch)
					summary.add_scalar(tag='content_from_content/test', scalar_value=score_test, global_step=epoch)

				score_train, score_test = self.classification_score(X=content_codes, y=classes)
				summary.add_scalar(tag='class_from_content/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_content/test', scalar_value=score_test, global_step=epoch)

				score_train, score_test = self.classification_score(X=style_codes, y=classes)
				summary.add_scalar(tag='class_from_style/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_style/test', scalar_value=score_test, global_step=epoch)

				if contents is not None:
					if contents.dtype == np.int64:
						score_train, score_test = self.classification_score(X=style_codes, y=contents)
					else:
						score_train, score_test = self.regression_score(X=style_codes, y=contents)

				summary.add_scalar(tag='content_from_style/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='content_from_style/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def adv_loss(self, logits, target):
		assert target in [1, 0]

		targets = torch.full_like(logits, fill_value=target)
		return torch.mean((logits - targets) ** 2)

		# loss = F.binary_cross_entropy_with_logits(logits, targets)
		# return loss

	# def gradient_penalty(self, d_out, x_in):
	# 	batch_size = x_in.size(0)
	# 	grad_dout = torch.autograd.grad(
	# 		outputs=d_out.sum(), inputs=x_in,
	# 		create_graph=True, retain_graph=True, only_inputs=True
	# 	)[0]
	# 	grad_dout2 = grad_dout.pow(2)
	# 	assert(grad_dout2.size() == x_in.size())
	# 	reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
	# 	return reg

	@torch.no_grad()
	def generate_samples(self, dataset, n_samples=5, randomized=False, style_only=False):
		self.generator.eval()
		self.style_encoder.eval()

		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		style_codes = self.style_encoder(samples['img'], samples['class_id'])

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				class_id_from = j if style_only else i
				out = self.generator(
					samples['img_id'][[j]], style_codes[[i]], samples['class_id'][[class_id_from]]
				)

				converted_imgs.append(out['img'][0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		return summary

	@staticmethod
	def regression_score(X, y):
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

		regressor = LinearRegression()
		regressor.fit(X_train, y_train)

		err_train = regressor.score(X_train, y_train)
		err_test = regressor.score(X_test, y_test)

		return err_train, err_test

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

	@torch.no_grad()
	def extract_codes(self, dataset):
		data_loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)

		content_codes = []
		style_codes = []

		for batch in data_loader:
			batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

			batch_content_codes = self.generator.content_embedding(batch['img_id'])
			batch_style_codes = self.style_encoder(batch['img'], batch['class_id'])

			content_codes.append(batch_content_codes.cpu().numpy())
			style_codes.append(batch_style_codes.cpu().numpy())

		content_codes = np.concatenate(content_codes, axis=0)
		style_codes = np.concatenate(style_codes, axis=0)

		return content_codes, style_codes
