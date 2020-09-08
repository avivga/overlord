import argparse
import os

import numpy as np

import data
from assets import AssetManager
from network.training import Model
from config import base_config


def preprocess(args, extras=[]):
	assets = AssetManager(args.base_dir)

	img_dataset_def = data.supported_datasets[args.dataset_id]
	img_dataset = img_dataset_def(args.dataset_path, extras)

	np.savez(file=assets.get_preprocess_file_path(args.data_name), **img_dataset.read())


def split(args):
	assets = AssetManager(args.base_dir)
	data = np.load(assets.get_preprocess_file_path(args.input_data_name))

	unique_values = np.unique(data[args.key])
	values_a = np.random.choice(unique_values, size=args.n_values, replace=False)
	values_b = np.array([v for v in unique_values if v not in values_a])
	idx_a = np.isin(data[args.key], values_a)
	idx_b = np.isin(data[args.key], values_b)

	arrays_a = {
		key: data[key][idx_a]
		for key in data.files
	}

	arrays_b = {
		key: data[key][idx_b]
		for key in data.files
	}

	reindex_a = np.zeros(shape=(unique_values.size,), dtype=arrays_a[args.key].dtype)
	reindex_a[values_a] = np.arange(args.n_values)
	arrays_a[args.key] = reindex_a[arrays_a[args.key]]

	reindex_b = np.zeros(shape=(unique_values.size,), dtype=arrays_b[args.key].dtype)
	reindex_b[values_b] = np.arange(unique_values.size - args.n_values)
	arrays_b[args.key] = reindex_b[arrays_b[args.key]]

	np.savez(file=assets.get_preprocess_file_path(args.output_data_name_a), **arrays_a)
	np.savez(file=assets.get_preprocess_file_path(args.output_data_name_b), **arrays_b)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	config = dict(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_classes=np.unique(classes).size
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

	amortized_model_dir = os.path.join(model_dir, 'amortized')
	if not os.path.exists(amortized_model_dir):
		os.mkdir(amortized_model_dir)

	amortized_tensorboard_dir = os.path.join(tensorboard_dir, 'amortized')
	if not os.path.exists(amortized_tensorboard_dir):
		os.mkdir(amortized_tensorboard_dir)

	model = Model.load(model_dir)
	model.amortize(imgs, classes, amortized_model_dir, amortized_tensorboard_dir)


def translate(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	eval_dir = assets.recreate_eval_dir(args.model_name)

	out_dir = os.path.join(eval_dir, 'translations')
	os.mkdir(out_dir)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	amortized_model_dir = os.path.join(model_dir, 'amortized')
	model = Model.load(amortized_model_dir)

	if args.full:
		model.translate_full(imgs, classes, args.n_translations_per_image, out_dir)
	else:
		model.translate(imgs, classes, args.n_translations_per_image, out_dir)


def summary(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	eval_dir = assets.get_eval_dir(args.model_name)

	out_dir = os.path.join(eval_dir, 'summaries')
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	amortized_model_dir = os.path.join(model_dir, 'amortized')
	model = Model.load(amortized_model_dir)
	model.summary(imgs, classes, args.n_summaries, args.summary_size, out_dir)


def encode(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	eval_dir = assets.get_eval_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['img'].astype(np.float32) / 255.0
	classes = data['class']

	if args.amortized:
		amortized_model_dir = os.path.join(model_dir, 'amortized')
		model = Model.load(amortized_model_dir)
		model.encode(imgs, classes, amortized=True, out_path=os.path.join(eval_dir, 'latents.npz'))
	else:
		model = Model.load(model_dir)
		model.encode(imgs, classes, amortized=False, out_path=os.path.join(eval_dir, 'embeddings.npz'))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=data.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	split_parser = action_parsers.add_parser('split')
	split_parser.add_argument('-i', '--input-data-name', type=str, required=True)
	split_parser.add_argument('-a', '--output-data-name-a', type=str, required=True)
	split_parser.add_argument('-b', '--output-data-name-b', type=str, required=True)
	split_parser.add_argument('-k', '--key', type=str, required=True)
	split_parser.add_argument('-n', '--n-values', type=int, required=True)
	split_parser.set_defaults(func=split)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.set_defaults(func=train)

	amortize_parser = action_parsers.add_parser('amortize')
	amortize_parser.add_argument('-dn', '--data-name', type=str, required=True)
	amortize_parser.add_argument('-mn', '--model-name', type=str, required=True)
	amortize_parser.set_defaults(func=amortize)

	translate_parser = action_parsers.add_parser('translate')
	translate_parser.add_argument('-dn', '--data-name', type=str, required=True)
	translate_parser.add_argument('-mn', '--model-name', type=str, required=True)
	translate_parser.add_argument('-nt', '--n-translations-per-image', type=int, required=True)
	translate_parser.add_argument('-f', '--full', action='store_true')
	translate_parser.set_defaults(func=translate)

	summary_parser = action_parsers.add_parser('summary')
	summary_parser.add_argument('-dn', '--data-name', type=str, required=True)
	summary_parser.add_argument('-mn', '--model-name', type=str, required=True)
	summary_parser.add_argument('-ns', '--n-summaries', type=int, required=True)
	summary_parser.add_argument('-ss', '--summary-size', type=int, required=True)
	summary_parser.set_defaults(func=summary)

	encode_parser = action_parsers.add_parser('encode')
	encode_parser.add_argument('-dn', '--data-name', type=str, required=True)
	encode_parser.add_argument('-mn', '--model-name', type=str, required=True)
	encode_parser.add_argument('-a', '--amortized', action='store_true')
	encode_parser.set_defaults(func=encode)

	args, extras = parser.parse_known_args()
	if len(extras) == 0:
		args.func(args)
	else:
		args.func(args, extras)


if __name__ == '__main__':
	main()
