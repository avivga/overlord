import argparse

import numpy as np

import dataset
from assets import AssetManager
from model.training import SLord
from config import base_config


def preprocess(args, extras=[]):
	assets = AssetManager(args.base_dir)

	img_dataset_def = dataset.supported_datasets[args.dataset_id]
	img_dataset = img_dataset_def(args.dataset_path, extras)

	np.savez(file=assets.get_preprocess_file_path(args.data_name), **img_dataset.read())


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = ((data['img'].astype(np.float32) / 255.0) * 2) - 1

	config = dict(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_classes=np.unique(data['class']).size
	)

	config.update(base_config)

	slord = SLord(config)
	slord.train_latent(imgs, data['class'], model_dir, tensorboard_dir)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.set_defaults(func=train)

	args, extras = parser.parse_known_args()
	if len(extras) == 0:
		args.func(args)
	else:
		args.func(args, extras)


if __name__ == '__main__':
	main()
