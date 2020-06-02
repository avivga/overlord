import argparse
import os
import glob
import imageio

import numpy as np
np.random.seed(1337)

from keras.models import Model, Sequential
from keras.layers import Layer, Concatenate, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers, losses, utils
from keras.applications import vgg19


def build(args):
	vgg_conv = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

	for layer in vgg_conv.layers[:-4]:
		layer.trainable = False

	model = Sequential()
	model.add(vgg_conv)

	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))

	model.compile(
		loss=losses.categorical_crossentropy,
		optimizer=optimizers.adam(lr=1e-4),
		metrics=['accuracy']
	)

	model.summary()
	return model


def read_dataset(args):
	imgs = []
	classes = []

	for i in range(args.n_classes):
		paths = glob.glob(os.path.join(args.eval_dir, '{}-to-*'.format(i), '*'))
		class_imgs = np.stack([imageio.imread(f) for f in paths], axis=0)
		class_ids = np.full(shape=(class_imgs.shape[0], ), fill_value=i)

		imgs.append(class_imgs)
		classes.append(class_ids)

	imgs = np.concatenate(imgs, axis=0)
	classes = np.concatenate(classes, axis=0)

	imgs = (imgs / 255.0) * 2 - 1
	return imgs, classes


def classify(args):
	imgs, classes = read_dataset(args)
	classifier = build(args)

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', min_delta=1e-3, factor=0.5, patience=5, verbose=1)
	early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-3, patience=10, verbose=1)

	n_samples = imgs.shape[0]
	n_samples_validation = int(args.validation_split * n_samples)
	validation_idx = np.random.choice(n_samples, size=n_samples_validation, replace=False)
	train_idx = ~np.isin(np.arange(n_samples), validation_idx)

	classifier.fit(
		x=imgs[train_idx], y=utils.to_categorical(classes[train_idx], args.n_classes),

		validation_data=(
			imgs[validation_idx], utils.to_categorical(classes[validation_idx], args.n_classes)
		),

		batch_size=64, epochs=1000,
		callbacks=[reduce_lr],
		verbose=1
	)

	train_loss, train_accuracy = classifier.evaluate(
		x=imgs[train_idx],
		y=utils.to_categorical(classes[train_idx], args.n_classes)
	)

	print('train accuracy: %f' % train_accuracy)

	validation_loss, validation_accuracy = classifier.evaluate(
		x=imgs[validation_idx],
		y=utils.to_categorical(classes[validation_idx], args.n_classes)
	)

	print('validation accuracy: %f' % validation_accuracy)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval-dir', type=str, required=True)
	parser.add_argument('--n-classes', type=int, default=3)
	parser.add_argument('--validation-split', type=float, default=0.2)

	args = parser.parse_args()
	classify(args)
