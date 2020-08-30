import argparse
import os

import numpy as np

import dataset
from assets import AssetManager
from network.training import Model
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
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	unique_classes = np.unique(classes)
	class_index = np.arange(np.max(unique_classes) + 1)
	class_index[unique_classes] = np.arange(unique_classes.size)
	classes = class_index[classes]

	config = dict(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_classes=unique_classes.size
	)

	config.update(base_config)

	model = Model(config)
	model.train(imgs, classes, model_dir, tensorboard_dir)


def amortize(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	tensorboard_dir = assets.get_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	unique_classes = np.unique(classes)
	class_index = np.arange(np.max(unique_classes) + 1)
	class_index[unique_classes] = np.arange(unique_classes.size)
	classes = class_index[classes]

	amortized_model_dir = os.path.join(model_dir, 'amortized')
	if not os.path.exists(amortized_model_dir):
		os.mkdir(amortized_model_dir)

	amortized_tensorboard_dir = os.path.join(tensorboard_dir, 'amortized')
	if not os.path.exists(amortized_tensorboard_dir):
		os.mkdir(amortized_tensorboard_dir)

	model = Model.load(model_dir)
	model.amortize(imgs, classes, amortized_model_dir, amortized_tensorboard_dir)


def evaluate(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	eval_dir = assets.recreate_eval_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	amortized_model_dir = os.path.join(model_dir, 'amortized')
	model = Model.load(amortized_model_dir)
	model.generate_for_evaluation(imgs, classes, args.n_translations_per_image, eval_dir)


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

	amortize_parser = action_parsers.add_parser('amortize')
	amortize_parser.add_argument('-dn', '--data-name', type=str, required=True)
	amortize_parser.add_argument('-mn', '--model-name', type=str, required=True)
	amortize_parser.set_defaults(func=amortize)

	evaluate_parser = action_parsers.add_parser('evaluate')
	evaluate_parser.add_argument('-dn', '--data-name', type=str, required=True)
	evaluate_parser.add_argument('-mn', '--model-name', type=str, required=True)
	evaluate_parser.add_argument('-nt', '--n-translations-per-image', type=int, required=True)
	evaluate_parser.set_defaults(func=evaluate)

	args, extras = parser.parse_known_args()
	if len(extras) == 0:
		args.func(args)
	else:
		args.func(args, extras)


if __name__ == '__main__':
	main()
