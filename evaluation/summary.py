import argparse
import os

import imageio
import numpy as np

import torch
from torchvision import utils

from model.training import SLord
from model.utils import NamedTensorDataset


@torch.no_grad()
def sample(args):
	os.makedirs(args.output_dir)

	slord = SLord.load(args.model_dir)
	slord.content_encoder.eval()
	slord.style_encoder.eval()
	slord.generator.eval()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	data = np.load(args.data_path)
	imgs = ((data['img'].astype(np.float32) / 255.0) * 2) - 1
	classes = data['class']

	data = dict(
		img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
		img_id=torch.from_numpy(np.arange(imgs.shape[0])),
		class_id=torch.from_numpy(classes.astype(np.int64))
	)

	dataset = NamedTensorDataset(data)

	random = np.random.RandomState(seed=1337)
	for s in range(args.n_summaries):
		img_idx = torch.from_numpy(random.choice(len(dataset), size=args.n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(device) for name, tensor in samples.items()}

		content_codes = slord.content_encoder(samples['img'])
		style_codes = slord.style_encoder(samples['img'], samples['class_id'])

		blank = torch.ones_like(samples['img'][0])
		summary = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(args.n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(args.n_samples):
				out = slord.generator(content_codes[[j]], style_codes[[i]])
				converted_imgs.append(out[0])

			summary.append(torch.cat(converted_imgs, dim=2))

		summary = torch.cat(summary, dim=1)
		summary = ((summary + 1) / 2).clamp(0, 1)

		utils.save_image(summary, os.path.join(args.output_dir, '{:04d}.png'.format(s)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-dir', type=str, required=True)
	parser.add_argument('--data-path', type=str, required=True)
	parser.add_argument('--output-dir', type=str, required=True)
	parser.add_argument('--n-samples', type=int, default=10)
	parser.add_argument('--n-summaries', type=int, default=5)

	args = parser.parse_args()
	sample(args)
