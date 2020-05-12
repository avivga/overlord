import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import Generator, MsImageDis
from model.utils import NamedTensorDataset


class SLord:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.generator = Generator(self.config)
		self.generator.init()
		self.generator.to(self.device)

		self.discriminator = MsImageDis()
		self.discriminator.to(self.device)

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		slord = SLord(config)
		slord.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))

		return slord

	def save(self, model_dir):
		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		torch.save(self.generator.state_dict(), os.path.join(model_dir, 'generator.pth'))

	def train_latent(self, imgs, classes, model_dir, tensorboard_dir):
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

		reconstruction_loss_fn = L1Loss()

		generator_optimizer = Adam([
			{
				'params': itertools.chain(
					self.generator.modulation.parameters(),
					self.generator.decoder.parameters()
				),

				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(
					self.generator.content_embedding.parameters(),
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

		# generator_scheduler = CosineAnnealingLR(
		# 	generator_optimizer,
		# 	T_max=self.config['train']['n_epochs'] * len(data_loader),
		# 	eta_min=self.config['train']['learning_rate']['min']
		# )
		#
		# discriminator_scheduler = CosineAnnealingLR(
		# 	discriminator_optimizer,
		# 	T_max=self.config['train']['n_epochs'] * len(data_loader),
		# 	eta_min=self.config['train']['learning_rate']['min']
		# )

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['train']['n_epochs']):
			self.generator.train()
			self.discriminator.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
				out = self.generator(batch['img_id'], batch['class_id'])

				discriminator_optimizer.zero_grad()
				loss_discriminator = self.discriminator.calc_dis_loss(out['img'].detach(), batch['img'])

				loss_discriminator.backward()
				discriminator_optimizer.step()
				# discriminator_scheduler.step()

				generator_optimizer.zero_grad()
				loss_reconstruction = reconstruction_loss_fn(out['img'], batch['img'])
				loss_content = torch.sum(out['content_code'] ** 2, dim=1).mean()
				loss_adversarial = self.discriminator.calc_gen_loss(out['img'])

				loss_generator = (
					self.config['train']['loss_weights']['reconstruction'] * loss_reconstruction
					+ self.config['train']['loss_weights']['content'] * loss_content
					+ self.config['train']['loss_weights']['adversarial'] * loss_adversarial
				)

				loss_generator.backward()
				generator_optimizer.step()
				# generator_scheduler.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(gen_loss=loss_generator.item(), disc_loss=loss_discriminator.item())

			pbar.close()

			summary.add_scalar(tag='loss-generator', scalar_value=loss_generator.item(), global_step=epoch)
			summary.add_scalar(tag='loss-discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)

			samples_fixed = self.generate_samples(dataset, randomized=False)
			samples_random = self.generate_samples(dataset, randomized=True)

			summary.add_image(tag='samples-fixed', img_tensor=samples_fixed, global_step=epoch)
			summary.add_image(tag='samples-random', img_tensor=samples_random, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def generate_samples(self, dataset, n_samples=5, randomized=False):
		self.generator.eval()

		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		output = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.generator(samples['img_id'][[j]], samples['class_id'][[i]])
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

		return torch.cat(output, dim=1)
