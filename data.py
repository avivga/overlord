import os
import argparse
from abc import ABC, abstractmethod
import glob
import random

import imageio
import numpy as np
import scipy.io
import cv2
import json

from xml.etree import ElementTree
from tqdm import tqdm


class DataSet(ABC):

	def __init__(self, base_dir=None, extras=None):
		super().__init__()
		self._base_dir = base_dir
		self._extras = extras

	@abstractmethod
	def read(self):
		pass


class AFHQ(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=256)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		class_ids = sorted(os.listdir(os.path.join(self._base_dir, self.split)))

		imgs = []
		classes = []

		for i, class_id in enumerate(class_ids):
			class_dir = os.path.join(self._base_dir, self.split, class_id)
			class_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
			imgs.append(np.stack([cv2.resize(imageio.imread(f), dsize=(self.img_size, self.img_size)) for f in class_paths], axis=0))
			classes.append(np.full((len(class_paths), ), fill_value=i, dtype=np.uint32))

		return {
			'img': np.concatenate(imgs, axis=0),
			'class': np.concatenate(classes, axis=0)
		}


class CelebAHQ(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-pr', '--parts', type=str, nargs='+', required=True)
		parser.add_argument('-is', '--img-size', type=int, default=256)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def __read_attributes(self):
		with open(os.path.join(self._base_dir, 'CelebAMask-HQ', 'CelebAMask-HQ-attribute-anno.txt'), 'r') as fp:
			lines = fp.read().splitlines()

		attribute_names = lines[1].split()
		attributes = dict()
		for line in lines[2:]:
			tokens = line.split()
			img_name = os.path.splitext(tokens[0])[0]
			img_attributes = np.array(list(map(int, tokens[1:])))
			img_attributes[img_attributes == -1] = 0
			attributes[img_name] = img_attributes

		return attributes, attribute_names

	def read(self):
		img_names = sorted(os.listdir(os.path.join(self._base_dir, 'x1024')))
		attributes_map, attribute_names = self.__read_attributes()

		mask_paths = glob.glob(os.path.join(self._base_dir, 'CelebAMask-HQ', 'CelebAMask-HQ-mask-anno', '*', '*.png'))
		masks_index = dict()
		for mask_path in mask_paths:
			mask_id = os.path.splitext(os.path.basename(mask_path))[0].split('_')[0]
			if mask_id not in masks_index:
				masks_index[mask_id] = list()

			masks_index[mask_id].append(mask_path)

		imgs = np.empty(shape=(len(img_names), self.img_size, self.img_size, 3), dtype=np.uint8)
		masks = np.empty(shape=(len(img_names), self.img_size, self.img_size), dtype=np.uint8)
		attributes = np.full(shape=(len(img_names), 40), fill_value=-1, dtype=np.int16)

		for i, img_name in enumerate(tqdm(img_names)):
			img_path = os.path.join(self._base_dir, 'x1024', img_name)
			img = imageio.imread(img_path)
			imgs[i] = cv2.resize(img, dsize=(self.img_size, self.img_size))

			img_id = os.path.splitext(img_name)[0]
			mask_id = '{:05d}'.format(int(img_id))

			masks[i] = np.zeros(shape=(self.img_size, self.img_size), dtype=np.uint8)
			for mask_path in masks_index[mask_id]:
				part = '_'.join(os.path.splitext(os.path.basename(mask_path))[0].split('_')[1:])
				if part in self.parts:
					mask = imageio.imread(mask_path)[..., 0]
					mask = mask // 255
					masks[i] = np.clip(masks[i] + cv2.resize(mask, dsize=(self.img_size, self.img_size)), a_min=0, a_max=1)

			attributes[i] = attributes_map[img_id]

		gender = attributes[:, attribute_names.index('Male')]

		return {
			'img': imgs,
			'mask': masks,
			'attributes': attributes,
			'class': gender
		}


class FFHQ(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-is', '--img-size', type=int, default=256)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		imgs = np.empty(shape=(70000, self.img_size, self.img_size, 3), dtype=np.uint8)
		age = np.full(shape=(70000,), fill_value=-1, dtype=np.float32)
		gender = np.full(shape=(70000,), fill_value=-1, dtype=np.int16)

		img_ids = np.arange(70000)
		for i in tqdm(img_ids):
			img_path = os.path.join(self._base_dir, 'imgs-x256', 'img{:08d}.png'.format(i))
			imgs[i] = cv2.resize(imageio.imread(img_path), dsize=(self.img_size, self.img_size))

			features_path = os.path.join(self._base_dir, 'features', '{:05d}.json'.format(i))
			with open(features_path, 'r') as features_fp:
				features = json.load(features_fp)
				if len(features) != 0:
					age[i] = features[0]['faceAttributes']['age']
					gender[i] = (features[0]['faceAttributes']['gender'] == 'male')

		valid_idx = (age != -1)

		return {
			'img': imgs[valid_idx],
			'age': age[valid_idx],
			'gender': gender[valid_idx],
			'class': age[valid_idx].astype(np.int16) // 10
		}


class Edges2Shoes(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=256)
		parser.add_argument('-ni', '--n-images', type=int, required=False)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		img_paths = glob.glob(os.path.join(self._base_dir, self.split, '*.jpg'))

		if self.n_images:
			img_paths = random.sample(img_paths, k=self.n_images)

		edge_imgs = np.empty(shape=(len(img_paths), self.img_size, self.img_size, 3), dtype=np.uint8)
		shoe_imgs = np.empty(shape=(len(img_paths), self.img_size, self.img_size, 3), dtype=np.uint8)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])
			img = cv2.resize(img, dsize=(self.img_size * 2, self.img_size))

			edge_imgs[i] = img[:, :self.img_size, :]
			shoe_imgs[i] = img[:, self.img_size:, :]

		imgs = np.concatenate((edge_imgs, shoe_imgs), axis=0)
		classes = np.concatenate((
			np.zeros(shape=(edge_imgs.shape[0], ), dtype=np.uint8),
			np.ones(shape=(shoe_imgs.shape[0], ), dtype=np.uint8)
		), axis=0)

		return {
			'img': imgs,
			'class': classes
		}


class CelebA(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir, extras)

		parser = argparse.ArgumentParser()
		parser.add_argument('-cs', '--crop-size', type=int, nargs=2, default=(128, 128))
		parser.add_argument('-ts', '--target-size', type=int, nargs=2, default=(128, 128))
		parser.add_argument('-ni', '--n-identities', type=int, required=False)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

		self.__imgs_dir = os.path.join(self._base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self._base_dir, 'Anno', 'identity_CelebA.txt')
		self.__attribute_map_path = os.path.join(self._base_dir, 'Anno', 'list_attr_celeba.txt')

	def __list_imgs(self):
		with open(self.__identity_map_path, 'r') as fd:
			lines = fd.read().splitlines()

		img_paths = []
		identities = []

		for line in lines:
			img_name, identity = line.split(' ')
			img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')

			img_paths.append(img_path)
			identities.append(identity)

		return img_paths, identities

	def __list_attributes(self):
		with open(self.__attribute_map_path, 'r') as fd:
			lines = fd.read().splitlines()[2:]

		attributes = dict()
		for line in lines:
			tokens = line.split()
			img_name = os.path.splitext(tokens[0])[0]
			img_attributes = np.array(list(map(int, tokens[1:])))
			img_attributes[img_attributes == -1] = 0
			attributes[img_name] = img_attributes

		return attributes

	def read(self):
		img_paths, identity_ids = self.__list_imgs()
		attritbute_map = self.__list_attributes()

		unique_identities = list(set(identity_ids))

		if self.n_identities:
			unique_identities = random.sample(unique_identities, k=self.n_identities)
			img_paths, identity_ids = zip(*[(path, identity) for path, identity in zip(img_paths, identity_ids) if identity in unique_identities])

		imgs = np.empty(shape=(len(img_paths), self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
		identities = np.empty(shape=(len(img_paths), ), dtype=np.int32)
		attributes = np.empty(shape=(len(img_paths), 40), dtype=np.int8)

		for i in tqdm(range(len(img_paths))):
			img_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
			img = imageio.imread(img_paths[i])

			img = img[
				(img.shape[0] // 2 - self.crop_size[0] // 2):(img.shape[0] // 2 + self.crop_size[0] // 2),
				(img.shape[1] // 2 - self.crop_size[1] // 2):(img.shape[1] // 2 + self.crop_size[1] // 2)
			]

			imgs[i] = cv2.resize(img, dsize=tuple(self.target_size))
			identities[i] = unique_identities.index(identity_ids[i])
			attributes[i] = attritbute_map[img_name]

		return {
			'img': imgs,
			'class': identities,
			'attributes': attributes
		}


supported_datasets = {
	'afhq': AFHQ,
	'celebahq': CelebAHQ,
	'ffhq': FFHQ,
	'edges2shoes': Edges2Shoes,
	'celeba': CelebA
}
