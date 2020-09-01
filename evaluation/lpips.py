# based on evaluation of StarGAN-v2
import argparse
import os
import glob
import imageio
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize(x, eps=1e-10):
	return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = models.alexnet(pretrained=True).features
		self.channels = []
		for layer in self.layers:
			if isinstance(layer, nn.Conv2d):
				self.channels.append(layer.out_channels)

	def forward(self, x):
		fmaps = []
		for layer in self.layers:
			x = layer(x)
			if isinstance(layer, nn.ReLU):
				fmaps.append(x)
		return fmaps


class Conv1x1(nn.Module):
	def __init__(self, in_channels, out_channels=1):
		super().__init__()
		self.main = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

	def forward(self, x):
		return self.main(x)


class LPIPS(nn.Module):
	def __init__(self):
		super().__init__()
		self.alexnet = AlexNet()
		self.lpips_weights = nn.ModuleList()
		for channels in self.alexnet.channels:
			self.lpips_weights.append(Conv1x1(channels, 1))
		self._load_lpips_weights()
		# imagenet normalization for range [-1, 1]
		self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
		self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

	def _load_lpips_weights(self):
		own_state_dict = self.state_dict()
		state_dict = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lpips_weights.ckpt'))

		for name, param in state_dict.items():
			if name in own_state_dict:
				own_state_dict[name].copy_(param)

	def forward(self, x, y):
		x = (x - self.mu) / self.sigma
		y = (y - self.mu) / self.sigma
		x_fmaps = self.alexnet(x)
		y_fmaps = self.alexnet(y)
		lpips_value = 0
		for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
			x_fmap = normalize(x_fmap)
			y_fmap = normalize(y_fmap)
			lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
		return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(lpips, group_of_images):
	# group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
	lpips_values = []
	num_rand_outputs = len(group_of_images)

	# calculate the average of pairwise distances among all random outputs
	for i in range(num_rand_outputs-1):
		for j in range(i+1, num_rand_outputs):
			lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
	lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
	return lpips_value.item()


def eval_lpips(args):
	lpips = LPIPS().eval().to(device)

	all_scores = {}
	translations_dir = os.path.join(args.eval_dir, 'translations')
	for direction in os.listdir(translations_dir):
		scores = []
		content_file_names = os.listdir(os.path.join(translations_dir, direction, 'content'))

		n_batches = int(np.ceil(len(content_file_names) / args.batch_size))

		pbar = tqdm(range(n_batches))
		pbar.set_description_str('lpips on {}'.format(direction))

		for b in pbar:
			content_ids = [
				os.path.splitext(f)[0].split('-')[0]
				for f in content_file_names[(b*args.batch_size):((b+1)*args.batch_size)]
			]

			translation_paths = {
				content_id: glob.glob(os.path.join(translations_dir, direction, 'translation', '{}-*.png'.format(content_id)))
				for content_id in content_ids
			}

			group_of_images = []
			for i in range(args.n_translations_per_image):
				imgs = np.stack([imageio.imread(translation_paths[content_id][i]) for content_id in content_ids], axis=0)
				imgs = ((imgs.astype(np.float32) / 255) * 2) - 1

				batch = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
				group_of_images.append(batch)

			score = calculate_lpips_given_images(lpips, group_of_images)
			scores.append(score)

		all_scores[direction] = np.mean(scores)

	all_scores['mean'] = np.mean(list(all_scores.values()))

	with open(os.path.join(args.eval_dir, 'lpips.json'), 'w') as f:
		json.dump(all_scores, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval-dir', type=str, required=True)
	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--n-translations-per-image', type=int, default=10)
	args = parser.parse_args()

	eval_lpips(args)
