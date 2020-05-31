import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import ContentEncoder, Generator, Discriminator, StyleEncoder, MappingNetwork
from model.utils import NamedTensorDataset


class SLord:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.content_encoder = ContentEncoder(self.config)
		self.content_encoder.to(self.device)

		self.generator = Generator(self.config)
		self.generator.to(self.device)

		self.discriminator = Discriminator(self.config)
		self.discriminator.to(self.device)

		self.style_encoder = StyleEncoder(self.config)
		self.style_encoder.to(self.device)

		self.mapping = MappingNetwork(self.config)
		self.mapping.to(self.device)

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		slord = SLord(config)
		slord.content_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'content_encoder.pth')))
		slord.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
		slord.discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth')))
		slord.style_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'style_encoder.pth')))
		slord.mapping.load_state_dict(torch.load(os.path.join(model_dir, 'mapping.pth')))

		return slord

	def save(self, model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		torch.save(self.content_encoder.state_dict(), os.path.join(model_dir, 'content_encoder.pth'))
		torch.save(self.generator.state_dict(), os.path.join(model_dir, 'generator.pth'))
		torch.save(self.discriminator.state_dict(), os.path.join(model_dir, 'discriminator.pth'))
		torch.save(self.style_encoder.state_dict(), os.path.join(model_dir, 'style_encoder.pth'))
		torch.save(self.mapping.state_dict(), os.path.join(model_dir, 'mapping.pth'))

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

		generator_optimizer = Adam(
			params=itertools.chain(self.content_encoder.parameters(), self.generator.parameters()),
			lr=self.config['train']['learning_rate']['generator'],
			betas=(0.0, 0.99), weight_decay=1e-4
		)

		discriminator_optimizer = Adam(
			params=self.discriminator.parameters(),
			lr=self.config['train']['learning_rate']['discriminator'],
			betas=(0.0, 0.99), weight_decay=1e-4
		)

		style_encoder_optimizer = Adam(
			params=self.style_encoder.parameters(),
			lr=self.config['train']['learning_rate']['style_encoder'],
			betas=(0.0, 0.99), weight_decay=1e-4
		)

		mapping_optimizer = Adam(
			params=self.mapping.parameters(),
			lr=self.config['train']['learning_rate']['mapping'],
			betas=(0.0, 0.99), weight_decay=1e-4
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['train']['n_epochs']):
			self.content_encoder.train()
			self.generator.train()
			self.discriminator.train()
			self.style_encoder.train()
			self.mapping.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				target_class_ids = np.random.choice(self.config['n_classes'], size=batch['img_id'].shape[0])
				batch['target_class_id'] = torch.from_numpy(target_class_ids).to(self.device)

				target_class_img_ids = np.array([np.random.choice(class_img_ids[class_id]) for class_id in target_class_ids])
				batch['target_class_img_id'] = torch.from_numpy(target_class_img_ids).to(self.device)
				batch['target_class_img'] = data['img'][target_class_img_ids].to(self.device)

				target_class_img_ids2 = np.array([np.random.choice(class_img_ids[class_id]) for class_id in target_class_ids])
				batch['target_class_img_id2'] = torch.from_numpy(target_class_img_ids2).to(self.device)
				batch['target_class_img2'] = data['img'][target_class_img_ids2].to(self.device)

				batch['target_class_z'] = torch.randn(batch['img_id'].shape[0], 16).to(self.device)
				batch['target_class_z2'] = torch.randn(batch['img_id'].shape[0], 16).to(self.device)

				loss_discriminator = self.do_discriminator(batch, sampling=True)
				discriminator_optimizer.zero_grad()
				loss_discriminator.backward()
				discriminator_optimizer.step()

				loss_discriminator = self.do_discriminator(batch, sampling=False)
				discriminator_optimizer.zero_grad()
				loss_discriminator.backward()
				discriminator_optimizer.step()

				losses_generator = self.do_generator(batch, sampling=True)
				loss_generator = 0
				for term, loss in losses_generator.items():
					loss_generator += self.config['train']['loss_weights'][term] * loss

				generator_optimizer.zero_grad()
				mapping_optimizer.zero_grad()
				style_encoder_optimizer.zero_grad()
				loss_generator.backward()
				generator_optimizer.step()
				mapping_optimizer.step()
				style_encoder_optimizer.step()

				losses_generator = self.do_generator(batch, sampling=False)
				loss_generator = 0
				for term, loss in losses_generator.items():
					loss_generator += self.config['train']['loss_weights'][term] * loss

				generator_optimizer.zero_grad()
				loss_generator.backward()
				generator_optimizer.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

			pbar.close()

			summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)
			summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)

			for term, loss in losses_generator.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			samples_fixed = self.generate_samples(dataset, randomized=False)
			samples_random = self.generate_samples(dataset, randomized=True)

			summary.add_image(tag='samples-fixed', img_tensor=samples_fixed, global_step=epoch)
			summary.add_image(tag='samples-random', img_tensor=samples_random, global_step=epoch)

			# styles_random = self.generate_samples(dataset, randomized=True, style_only=True)
			# summary.add_image(tag='styles-random', img_tensor=styles_random, global_step=epoch)

			if epoch % 5 == 0:
				content_codes = self.extract_codes(dataset)
				score_train, score_test = self.classification_score(X=content_codes, y=classes)
				summary.add_scalar(tag='class_from_content/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_content/test', scalar_value=score_test, global_step=epoch)

			# 	content_codes, style_codes = self.extract_codes(dataset)
			#
			# 	if contents is not None:
			# 		if contents.dtype == np.int64:
			# 			score_train, score_test = self.classification_score(X=content_codes, y=contents)
			# 		else:
			# 			score_train, score_test = self.regression_score(X=content_codes, y=contents)
			#
			# 		summary.add_scalar(tag='content_from_content/train', scalar_value=score_train, global_step=epoch)
			# 		summary.add_scalar(tag='content_from_content/test', scalar_value=score_test, global_step=epoch)
			#
			# 	score_train, score_test = self.classification_score(X=content_codes, y=classes)
			# 	summary.add_scalar(tag='class_from_content/train', scalar_value=score_train, global_step=epoch)
			# 	summary.add_scalar(tag='class_from_content/test', scalar_value=score_test, global_step=epoch)
			#
			# 	score_train, score_test = self.classification_score(X=style_codes, y=classes)
			# 	summary.add_scalar(tag='class_from_style/train', scalar_value=score_train, global_step=epoch)
			# 	summary.add_scalar(tag='class_from_style/test', scalar_value=score_test, global_step=epoch)
			#
			# 	if contents is not None:
			# 		if contents.dtype == np.int64:
			# 			score_train, score_test = self.classification_score(X=style_codes, y=contents)
			# 		else:
			# 			score_train, score_test = self.regression_score(X=style_codes, y=contents)
			#
			# 	summary.add_scalar(tag='content_from_style/train', scalar_value=score_train, global_step=epoch)
			# 	summary.add_scalar(tag='content_from_style/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def do_discriminator(self, batch, sampling=False):
		if sampling:
			style_code_target = self.mapping(batch['target_class_z'], batch['target_class_id'])
		else:
			style_code_target = self.style_encoder(batch['target_class_img'], batch['target_class_id'])

		with torch.no_grad():
			content_code = self.content_encoder(batch['img'])
			generated_img = self.generator(content_code, style_code_target)

		batch['img'].requires_grad_()  # for gradient penalty
		discriminator_fake = self.discriminator(generated_img, batch['target_class_id'])
		discriminator_real = self.discriminator(batch['img'], batch['class_id'])

		loss_fake = self.adv_loss(discriminator_fake, 0)
		loss_real = self.adv_loss(discriminator_real, 1)
		loss_gp = self.gradient_penalty(discriminator_real, batch['img'])

		loss_discriminator = loss_fake + loss_real + self.config['train']['loss_weights']['gradient_penalty'] * loss_gp
		return loss_discriminator

	def do_generator(self, batch, sampling=False):
		if sampling:
			style_code_target = self.mapping(batch['target_class_z'], batch['target_class_id'])
		else:
			style_code_target = self.style_encoder(batch['target_class_img'], batch['target_class_id'])

		content_code = self.content_encoder(batch['img'])
		generated_img = self.generator(content_code, style_code_target)

		discriminator_fake = self.discriminator(generated_img, batch['target_class_id'])
		loss_adversarial = self.adv_loss(discriminator_fake, 1)

		style_code_reconstructed = self.style_encoder(generated_img, batch['target_class_id'])
		loss_style_reconstruction = torch.mean(torch.abs(style_code_reconstructed - style_code_target))

		if sampling:
			style_code_target2 = self.mapping(batch['target_class_z2'], batch['target_class_id'])
		else:
			style_code_target2 = self.style_encoder(batch['target_class_img2'], batch['target_class_id'])

		with torch.no_grad():
			generated_img2 = self.generator(content_code, style_code_target2)

		loss_diversity = torch.mean(torch.abs(generated_img - generated_img2))

		style_code_original = self.style_encoder(batch['img'], batch['class_id'])
		reconstructed_img = self.generator(content_code, style_code_original)
		loss_reconstruction = torch.mean(torch.abs(reconstructed_img - batch['img']))

		loss_content_decay = torch.sum(content_code ** 2).mean()
		loss_style_decay = torch.sum(style_code_target ** 2).mean()

		return {
			'reconstruction': loss_reconstruction,
			'content_decay': loss_content_decay,
			'style_decay': loss_style_decay,
			'adversarial': loss_adversarial,
			'style_reconstruction': loss_style_reconstruction,
			'diversity': -loss_diversity
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
	def generate_samples(self, dataset, n_samples=10, randomized=False):
		self.content_encoder.eval()
		self.style_encoder.eval()
		self.generator.eval()

		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		content_codes = self.content_encoder(samples['img'])
		style_codes = self.style_encoder(samples['img'], samples['class_id'])

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.generator(content_codes[[j]], style_codes[[i]])
				converted_imgs.append(out[0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		summary = ((summary + 1) / 2).clamp(0, 1)
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
		self.content_encoder.eval()

		data_loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)

		content_codes = []
		for batch in data_loader:
			batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

			batch_content_codes = self.content_encoder(batch['img']).view((batch['img'].shape[0], -1))
			content_codes.append(batch_content_codes.cpu().numpy())

		content_codes = np.concatenate(content_codes, axis=0)
		return content_codes
