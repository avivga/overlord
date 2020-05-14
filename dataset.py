import os
import argparse
from abc import ABC, abstractmethod

import imageio
import numpy as np
import scipy.io
import cv2

from dataset_pascal import Pascal


class DataSet(ABC):

	def __init__(self, base_dir=None, extras=None):
		super().__init__()
		self._base_dir = base_dir
		self._extras = extras

	@abstractmethod
	def read(self):
		pass


class Cars3D(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir, extras)

		self.__data_path = os.path.join(base_dir, 'cars3d.npz')

	def read(self):
		imgs = np.load(self.__data_path)['imgs']
		classes = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)
		contents = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)

		for elevation in range(4):
			for azimuth in range(24):
				for object_id in range(183):
					img_idx = elevation * 24 * 183 + azimuth * 183 + object_id

					classes[img_idx] = object_id
					contents[img_idx] = elevation * 24 + azimuth

		return {
			'img': imgs,
			'class': classes
		}


class Cub(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=256)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		data = scipy.io.loadmat(os.path.join(self._base_dir, 'from_cmr', 'data', '{}_cub_cleaned.mat'.format(self.split)), struct_as_record=False, squeeze_me=True)

		n_images = data['images'].shape[0]
		imgs = []
		categories = []

		for i in range(n_images):
			img_struct = data['images'][i]
			img_path = os.path.join(self._base_dir, 'images', img_struct.rel_path)

			img = imageio.imread(img_path)
			if len(img.shape) == 2:
				img = np.tile(img[..., np.newaxis], reps=(1, 1, 3))

			img_masked = img * img_struct.mask[..., np.newaxis]
			bbox = dict(
				x1=img_struct.bbox.x1 - 1, x2=img_struct.bbox.x2 - 1,
				y1=img_struct.bbox.y1 - 1, y2=img_struct.bbox.y2 - 1
			)

			height = bbox['y2'] - bbox['y1'] + 1
			width = bbox['x2'] - bbox['x1'] + 1
			box_length = max(width, height)

			y1 = max(bbox['y1'] - (box_length - height) // 2, 0)
			y2 = y1 + box_length - 1

			x1 = max(bbox['x1'] - (box_length - width) // 2, 0)
			x2 = x1 + box_length - 1

			img_cropped = img_masked[y1:y2, x1:x2]
			imgs.append(cv2.resize(img_cropped, dsize=(self.img_size, self.img_size)))

			category_id = img_struct.rel_path.split('/')[0]
			categories.append(category_id)

		unique_categories = list(set(categories))
		categories = list(map(lambda c: unique_categories.index(c), categories))

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.array(categories)
		}


class Pascal3D(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-ct', '--category', type=str, required=True)
		parser.add_argument('-cp', '--classes-path', type=str, required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		pascal = Pascal(directory=self._base_dir, class_ids=[self.category], set_name=self.split)

		imgs = pascal.images_ref[self.category].transpose(0, 2, 3, 1)[..., :3]
		imgs = imgs[:, :, ::-1, :]
		imgs = [cv2.resize(imgs[i], dsize=(self.img_size, self.img_size)) for i in range(imgs.shape[0])]

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.load(self.classes_path)['classes']
		}


supported_datasets = {
	'cars3d': Cars3D,
	'cub': Cub,
	'pascal3d': Pascal3D
}
