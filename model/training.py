import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import Generator, Discriminator
from model.utils import NamedTensorDataset


class Model:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.content_embedding = nn.Embedding(num_embeddings=config['n_imgs'], embedding_dim=config['content_depth'] * 16 * 16)
		self.class_embedding = nn.Embedding(num_embeddings=config['n_classes'], embedding_dim=config['class_depth'] * 16 * 16)

		nn.init.uniform_(self.content_embedding.weight, a=-0.05, b=0.05)
		nn.init.uniform_(self.class_embedding.weight, a=-0.05, b=0.05)

		self.generator = Generator(self.config)
		self.generator.to(self.device)

		self.discriminator = Discriminator(self.config)
		self.discriminator.to(self.device)

		self.generator_optimizer = Adam([
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
			], betas=(0.5, 0.999)
		)

		self.discriminator_optimizer = Adam(
			params=self.discriminator.parameters(),
			lr=self.config['train']['learning_rate']['discriminator'],
			betas=(0.5, 0.999)
		)

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		model = Model(config)
		model.content_embedding.load_state_dict(torch.load(os.path.join(model_dir, 'content_embedding.pth')))
		model.class_embedding.load_state_dict(torch.load(os.path.join(model_dir, 'class_embedding.pth')))
		model.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
		model.discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth')))

		return model

	def save(self, model_dir, iteration):
		checkpoint_dir = os.path.join(model_dir, '{:08d}'.format(iteration))
		os.mkdir(checkpoint_dir)

		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		torch.save(self.content_embedding.state_dict(), os.path.join(checkpoint_dir, 'content_embedding.pth'))
		torch.save(self.class_embedding.state_dict(), os.path.join(checkpoint_dir, 'class_embedding.pth'))
		torch.save(self.generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
		torch.save(self.discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))

	def clone(self):
		model = Model(self.config)

		model.content_embedding.load_state_dict(self.content_embedding.state_dict())
		model.class_embedding.load_state_dict(self.class_embedding.state_dict())
		model.generator.load_state_dict(self.generator.state_dict())
		model.discriminator.load_state_dict(self.discriminator.state_dict())

		return model

	def train(self, imgs, classes, model_dir, tensorboard_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=False
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		for epoch in range(self.config['train']['n_epochs']):
			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				target_class_ids = np.random.choice(self.config['n_classes'], size=batch['img_id'].shape[0])
				batch['target_class_id'] = torch.from_numpy(target_class_ids)

				batch['content_code'] = self.content_embedding(batch['img_id'])
				batch['class_code'] = self.class_embedding(batch['class_id'])
				batch['target_class_code'] = self.class_embedding(batch['target_class_id'])

				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				self.generator.train()
				self.discriminator.train()

				losses_discriminator = self.do_discriminator(batch)
				loss_discriminator = (
					losses_discriminator['fake']
					+ losses_discriminator['real']
					+ self.config['train']['loss_weights']['gradient_penalty'] * losses_discriminator['gradient_penalty']
				)

				self.generator_optimizer.zero_grad()
				self.discriminator_optimizer.zero_grad()
				loss_discriminator.backward()
				self.discriminator_optimizer.step()

				losses_generator = self.do_generator(batch)
				loss_generator = 0
				for term, loss in losses_generator.items():
					loss_generator += self.config['train']['loss_weights'][term] * loss

				self.generator_optimizer.zero_grad()
				self.discriminator_optimizer.zero_grad()
				loss_generator.backward()
				self.generator_optimizer.step()

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

			if epoch % 5 == 0:
				content_codes = self.extract_codes(dataset)
				score_train, score_test = self.classification_score(X=content_codes, y=classes)
				summary.add_scalar(tag='class_from_content/train', scalar_value=score_train, global_step=epoch)
				summary.add_scalar(tag='class_from_content/test', scalar_value=score_test, global_step=epoch)

			self.save(model_dir, epoch)

		summary.close()

	def do_discriminator(self, batch):
		with torch.no_grad():
			out = self.generator(batch['content_code'], batch['target_class_code'])

		batch['img'].requires_grad_()  # for gradient penalty
		discriminator_fake = self.discriminator(out['img'], batch['target_class_id'])
		discriminator_real = self.discriminator(batch['img'], batch['class_id'])

		loss_fake = self.adv_loss(discriminator_fake, 0)
		loss_real = self.adv_loss(discriminator_real, 1)
		loss_gp = self.gradient_penalty(discriminator_real, batch['img'])

		return {
			'fake': loss_fake,
			'real': loss_real,
			'gradient_penalty': loss_gp
		}

	def do_generator(self, batch):
		out = self.generator(batch['content_code'], batch['target_class_code'])

		discriminator_fake = self.discriminator(out['img'], batch['target_class_id'])
		loss_adversarial = self.adv_loss(discriminator_fake, 1)

		out_reconstruction = self.generator(batch['content_code'], batch['class_code'])
		loss_reconstruction = torch.mean(torch.abs(out_reconstruction['img'] - batch['img']))

		loss_content_decay = torch.sum(batch['content_code'] ** 2, dim=1).mean()

		return {
			'reconstruction': loss_reconstruction,
			'content_decay': loss_content_decay,
			'adversarial': loss_adversarial
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
		self.generator.eval()

		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples['content_code'] = self.content_embedding(samples['img_id'])
		samples['class_code'] = self.class_embedding(samples['class_id'])
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.generator(samples['content_code'][[j]], samples['class_code'][[i]])
				converted_imgs.append(out['img'][0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		summary = ((summary + 1) / 2).clamp(0, 1)
		return summary

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
		for batch in data_loader:
			batch_content_codes = self.content_embedding(batch['img_id']).view((batch['img_id'].shape[0], -1))
			content_codes.append(batch_content_codes.numpy())

		content_codes = np.concatenate(content_codes, axis=0)
		return content_codes
