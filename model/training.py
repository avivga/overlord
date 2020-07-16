import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import ContentEncoder, Generator, Discriminator
from model.utils import NamedTensorDataset, he_init


class Model:

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

		# TODO: class embedding lr

		self.generator_optimizer = Adam(
			params=itertools.chain(self.content_encoder.parameters(), self.generator.parameters()),
			lr=self.config['train']['learning_rate']['generator'],
			betas=(0.0, 0.99), weight_decay=1e-4
		)

		self.discriminator_optimizer = Adam(
			params=self.discriminator.parameters(),
			lr=self.config['train']['learning_rate']['discriminator'],
			betas=(0.0, 0.99), weight_decay=1e-4
		)

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		model = Model(config)
		model.content_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'content_encoder.pth')))
		model.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
		model.discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth')))

		return model

	def save(self, model_dir, iteration):
		checkpoint_dir = os.path.join(model_dir, '{:08d}'.format(iteration))
		os.mkdir(checkpoint_dir)

		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		torch.save(self.content_encoder.state_dict(), os.path.join(checkpoint_dir, 'content_encoder.pth'))
		torch.save(self.generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
		torch.save(self.discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))

	def clone(self):
		model = Model(self.config)

		model.content_encoder.load_state_dict(self.content_encoder.state_dict())
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

		model_ema = self.clone()

		self.content_encoder.apply(he_init)
		self.generator.apply(he_init)
		self.discriminator.apply(he_init)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		iteration = 0
		for epoch in range(self.config['train']['n_epochs']):
			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				target_class_ids = np.random.choice(self.config['n_classes'], size=batch['img_id'].shape[0])
				batch['target_class_id'] = torch.from_numpy(target_class_ids).to(self.device)

				self.content_encoder.train()
				self.generator.train()
				self.discriminator.train()

				loss_discriminator = self.do_discriminator(batch)
				self.reset_grads()
				loss_discriminator.backward()
				self.discriminator_optimizer.step()

				losses_generator = self.do_generator(batch)
				loss_generator = 0
				for term, loss in losses_generator.items():
					loss_generator += self.config['train']['loss_weights'][term] * loss

				self.reset_grads()
				loss_generator.backward()
				self.generator_optimizer.step()

				model_ema.update_from(self)

				pbar.set_description_str('epoch #{} [{}]'.format(epoch, iteration))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

				if iteration % 1000 == 0:
					summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=iteration)
					summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=iteration)

					for term, loss in losses_generator.items():
						summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=iteration)

					samples_fixed = model_ema.generate_samples(dataset, randomized=False)
					samples_random = model_ema.generate_samples(dataset, randomized=True)

					summary.add_image(tag='samples-fixed', img_tensor=samples_fixed, global_step=iteration)
					summary.add_image(tag='samples-random', img_tensor=samples_random, global_step=iteration)

					model_ema.save(model_dir, iteration)

				iteration += 1

			pbar.close()

		summary.close()

	def do_discriminator(self, batch):
		with torch.no_grad():
			content_code = self.content_encoder(batch['img'])
			generated_img = self.generator(content_code, batch['target_class_id'])

		batch['img'].requires_grad_()  # for gradient penalty
		discriminator_fake = self.discriminator(generated_img, batch['target_class_id'])
		discriminator_real = self.discriminator(batch['img'], batch['class_id'])

		loss_fake = self.adv_loss(discriminator_fake, 0)
		loss_real = self.adv_loss(discriminator_real, 1)
		loss_gp = self.gradient_penalty(discriminator_real, batch['img'])

		loss_discriminator = loss_fake + loss_real + self.config['train']['loss_weights']['gradient_penalty'] * loss_gp
		return loss_discriminator

	def do_generator(self, batch):
		content_code = self.content_encoder(batch['img'])
		generated_img = self.generator(content_code, batch['target_class_id'])

		discriminator_fake = self.discriminator(generated_img, batch['target_class_id'])
		loss_adversarial = self.adv_loss(discriminator_fake, 1)

		content_code_fake = self.content_encoder(generated_img)
		reconstructed_img = self.generator(content_code_fake, batch['class_id'])
		loss_reconstruction = torch.mean(torch.abs(reconstructed_img - batch['img']))

		loss_content_decay = torch.sum(content_code ** 2, dim=[1, 2, 3]).mean()

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

	def reset_grads(self):
		self.generator_optimizer.zero_grad()
		self.discriminator_optimizer.zero_grad()

	@torch.no_grad()
	def generate_samples(self, dataset, n_samples=10, randomized=False):
		self.content_encoder.eval()
		self.generator.eval()

		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		content_codes = self.content_encoder(samples['img'])

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.generator(content_codes[[j]], samples['class_id'][[i]])
				converted_imgs.append(out[0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		summary = ((summary + 1) / 2).clamp(0, 1)
		return summary

	def update_from(self, other, beta=0.999):
		pairs = [
			(self.content_encoder, other.content_encoder),
			(self.generator, other.generator)
		]

		for model_ema, model in pairs:
			for param_ema, param in zip(model_ema.parameters(), model.parameters()):
				param_ema.data = torch.lerp(param.data, param_ema.data, beta)
