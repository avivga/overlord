import argparse
import os
import yaml
import imageio

import numpy as np

import data
from assets import AssetManager
from network.training import Model


def preprocess(args, extras=[]):
	assets = AssetManager(args.base_dir)

	img_dataset_def = data.supported_datasets[args.dataset_id]
	img_dataset = img_dataset_def(args.dataset_path, extras)

	np.savez(file=assets.get_preprocess_file_path(args.out_data_name), **img_dataset.read())


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	with open(os.path.join(os.path.dirname(__file__), 'config', '{}.yaml'.format(args.config)), 'r') as config_fp:
		config = yaml.safe_load(config_fp)

	assert config['correlation'] in [None, 'pose-invariant', 'localized']

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs']
	labels = data[config['labeled_attribute']]
	masks = data['masks'] if config['correlation'] == 'localized' else None

	config.update({
		'img_shape': imgs.shape[1:],
		'n_imgs': imgs.shape[0],
		'n_labels': np.unique(labels).size
	})

	model = Model(config)
	model.train_latent_model(imgs, labels, masks, model_dir, tensorboard_dir)
	model.warmup_amortized_model(imgs, labels, masks, model_dir, tensorboard_dir=os.path.join(tensorboard_dir, 'amortization'))
	model.tune_amortized_model(imgs, labels, masks, model_dir, tensorboard_dir=os.path.join(tensorboard_dir, 'synthesis'))


def manipulate(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	model = Model.load(model_dir)

	img = imageio.imread(args.img_path)
	if args.reference_img_path:
		reference_img = imageio.imread(args.reference_img_path)
		manipulated_img = model.manipulate_by_reference(img, reference_img)
	else:
		manipulated_img = model.manipulate_by_labels(img)

	imageio.imwrite(args.output_img_path, manipulated_img)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=data.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-odn', '--out-data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-cf', '--config', type=str, required=True)
	train_parser.set_defaults(func=train)

	manipulate_parser = action_parsers.add_parser('manipulate')
	manipulate_parser.add_argument('-mn', '--model-name', type=str, required=True)
	manipulate_parser.add_argument('-i', '--img-path', type=str, required=True)
	manipulate_parser.add_argument('-r', '--reference-img-path', type=str, required=False)
	manipulate_parser.add_argument('-o', '--output-img-path', type=str, required=True)
	manipulate_parser.set_defaults(func=manipulate)

	args, extras = parser.parse_known_args()
	if len(extras) == 0:
		args.func(args)
	else:
		args.func(args, extras)


if __name__ == '__main__':
	main()
