import argparse
import os
import json

from collections import OrderedDict
from tqdm import tqdm

import numpy as np
np.random.seed(1337)

import torch
torch.manual_seed(1337)

from torch.utils.data import DataLoader, TensorDataset
import torchvision

from model.training import SLord
from evaluation import lpips, fid


def load_data(path):
	data = np.load(path)

	imgs = ((data['img'].astype(np.float32) / 255.0) * 2) - 1
	classes = data['class']

	data = dict(
		img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
		img_id=torch.from_numpy(np.arange(imgs.shape[0])),
		class_id=torch.from_numpy(classes.astype(np.int64))
	)

	return data


@torch.no_grad()
def evaluate(args):
	os.makedirs(args.eval_dir)

	slord = SLord.load(args.model_dir)
	slord.content_encoder.eval()
	slord.style_encoder.eval()
	slord.generator.eval()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	data_train = load_data(args.train_data_path)
	data_val = load_data(args.val_data_path)

	domains = np.unique(data_train['class_id'].numpy())

	lpips_dict = OrderedDict()
	for trg_idx, trg_domain in enumerate(domains):
		src_domains = [x for x in domains if x != trg_domain]

		if args.mode == 'reference':
			dataset_target = TensorDataset(data_val['img'][data_val['class_id'] == trg_domain])
			data_loader_target = DataLoader(
				dataset_target, batch_size=args.batch_size,
				shuffle=False, pin_memory=True, drop_last=True
			)

		for src_idx, src_domain in enumerate(src_domains):
			dataset_source = TensorDataset(data_val['img'][data_val['class_id'] == src_domain])
			data_loader_source = DataLoader(
				dataset_source, batch_size=args.batch_size,
				shuffle=False, pin_memory=True, drop_last=True
			)

			task = '%s-to-%s' % (src_domain, trg_domain)
			path_fake = os.path.join(args.eval_dir, task)
			os.mkdir(path_fake)

			lpips_values = []
			print('Generating images and calculating LPIPS for %s...' % task)
			for i, batch in enumerate(tqdm(data_loader_source, total=len(data_loader_source))):
				x_src = batch[0]

				N = x_src.size(0)
				x_src = x_src.to(device)
				y_trg = torch.tensor([trg_idx] * N).to(device)

				group_of_images = []
				for j in range(args.num_outs_per_domain):
					if args.mode == 'latent':
						z_trg = torch.randn(N, args.latent_dim).to(device)
						s_trg = slord.mapping(z_trg, y_trg)
					else:
						try:
							x_ref = next(iter_ref)[0].to(device)
						except:
							iter_ref = iter(data_loader_target)
							x_ref = next(iter_ref)[0].to(device)

						if x_ref.size(0) > N:
							x_ref = x_ref[:N]

						s_trg = slord.style_encoder(x_ref, y_trg)

					c_src = slord.content_encoder(x_src)
					x_fake = slord.generator(c_src, s_trg)
					group_of_images.append(x_fake)

					# save generated images to calculate FID later
					for k in range(N):
						filename = os.path.join(path_fake, '%.4i_%.2i.png' % (i*args.batch_size+(k+1), j+1))
						save_image(x_fake[k], ncol=1, filename=filename)

				lpips_value = lpips.calculate_lpips_given_images(group_of_images)
				lpips_values.append(lpips_value)

			# calculate LPIPS for each task (e.g. cat2dog, dog2cat)
			lpips_mean = np.array(lpips_values).mean()
			lpips_dict['LPIPS_%s/%s' % (args.mode, task)] = lpips_mean

	# calculate the average LPIPS for all tasks
	lpips_mean = 0
	for _, value in lpips_dict.items():
		lpips_mean += value / len(lpips_dict)
	lpips_dict['LPIPS_%s/mean' % args.mode] = lpips_mean

	filename = os.path.join(args.eval_dir, 'LPIPS_%s.json' % args.mode)
	with open(filename, 'w') as f:
		json.dump(lpips_dict, f, indent=4, sort_keys=False)

	# calculate and report fid values
	calculate_fid_for_all_tasks(args, data_train, domains, mode=args.mode)


def calculate_fid_for_all_tasks(args, data_train, domains, mode):
	print('Calculating FID for all tasks...')
	fid_values = OrderedDict()
	for trg_domain in domains:
		src_domains = [x for x in domains if x != trg_domain]

		imgs_real = data_train['img'][data_train['class_id'] == trg_domain]
		for src_domain in src_domains:
			task = '%s-to-%s' % (src_domain, trg_domain)
			path_fake = os.path.join(args.eval_dir, task)

			print('Calculating FID for %s...' % task)
			fid_value = fid.calculate_fid_given_paths(
				imgs_real, path_fake,
				img_size=args.img_size,
				batch_size=args.batch_size)
			fid_values['FID_%s/%s' % (mode, task)] = fid_value

	# calculate the average FID for all tasks
	fid_mean = 0
	for _, value in fid_values.items():
		fid_mean += value / len(fid_values)
	fid_values['FID_%s/mean' % mode] = fid_mean

	# report FID values
	filename = os.path.join(args.eval_dir, 'FID_%s.json' % args.mode)
	with open(filename, 'w') as f:
		json.dump(fid_values, f, indent=4, sort_keys=False)


def save_image(x, ncol, filename):
	x = (x + 1) / 2
	x = x.clamp_(0, 1)

	torchvision.utils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-dir', type=str, required=True)
	parser.add_argument('--train-data-path', type=str, required=True)
	parser.add_argument('--val-data-path', type=str, required=True)
	parser.add_argument('--eval-dir', type=str, required=True)
	parser.add_argument('--mode', type=str, choices=['reference', 'latent'], required=True)
	parser.add_argument('--latent-dim', type=int, default=16)
	parser.add_argument('--img-size', type=int, default=128)
	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--num-outs-per-domain', type=int, default=10)

	args = parser.parse_args()
	evaluate(args)
