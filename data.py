import os
import argparse
import glob
import random
from abc import ABC, abstractmethod

import PIL
import numpy as np
import json

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
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'test'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=256)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		domains = sorted(os.listdir(os.path.join(self._base_dir, self.split)))

		imgs = []
		domain_ids = []

		for i, domain in enumerate(domains):
			domain_dir = os.path.join(self._base_dir, self.split, domain)
			domain_img_paths = [os.path.join(domain_dir, f) for f in os.listdir(domain_dir)]

			for f in tqdm(domain_img_paths):
				img = PIL.Image.open(f).resize(size=(self.img_size, self.img_size), resample=PIL.Image.BICUBIC)
				imgs.append(np.array(img))
				domain_ids.append(i)

		return {
			'imgs': np.stack(imgs, axis=0),
			'domain': np.array(domain_ids, dtype=np.int16)
		}


class CelebAHQ(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-pr', '--part', type=str, default='hair')
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

			img = PIL.Image.open(img_path)
			imgs[i] = np.array(img.resize(size=(self.img_size, self.img_size), resample=PIL.Image.BICUBIC))

			img_id = os.path.splitext(img_name)[0]
			mask_id = '{:05d}'.format(int(img_id))

			masks[i] = np.zeros(shape=(self.img_size, self.img_size), dtype=np.uint8)
			for mask_path in masks_index[mask_id]:
				part = '_'.join(os.path.splitext(os.path.basename(mask_path))[0].split('_')[1:])
				if part == self.part:
					mask = PIL.Image.open(mask_path)
					mask = np.array(mask.resize(size=(self.img_size, self.img_size), resample=PIL.Image.BICUBIC))
					masks[i] = mask[..., 0] // 255
					break

			attributes[i] = attributes_map[img_id]

		gender = attributes[:, attribute_names.index('Male')]

		return {
			'imgs': imgs,
			'masks': masks,
			'gender': gender
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

		img_ids = np.arange(70000)
		for i in tqdm(img_ids):
			img_path = os.path.join(self._base_dir, 'imgs-x256', 'img{:08d}.png'.format(i))

			img = PIL.Image.open(img_path)
			imgs[i] = np.array(img.resize(size=(self.img_size, self.img_size), resample=PIL.Image.BICUBIC))

			features_path = os.path.join(self._base_dir, 'features', '{:05d}.json'.format(i))
			with open(features_path, 'r') as features_fp:
				features = json.load(features_fp)
				if len(features) != 0:
					age[i] = features[0]['faceAttributes']['age']

		valid_idx = (age != -1)

		return {
			'imgs': imgs[valid_idx],
			'age': age[valid_idx].astype(np.int16) // 10
		}


class CelebA(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir, extras)

		parser = argparse.ArgumentParser()
		parser.add_argument('-cs', '--crop-size', type=int, default=128)
		parser.add_argument('-is', '--img-size', type=int, default=128)
		parser.add_argument('-ni', '--n-identities', type=int, required=False)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

		self.__imgs_dir = os.path.join(self._base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self._base_dir, 'Anno', 'identity_CelebA.txt')

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

	def read(self):
		img_paths, identity_ids = self.__list_imgs()
		unique_identities = list(set(identity_ids))

		if self.n_identities:
			unique_identities = random.sample(unique_identities, k=self.n_identities)
			img_paths, identity_ids = zip(*[(path, identity) for path, identity in zip(img_paths, identity_ids) if identity in unique_identities])

		imgs = np.empty(shape=(len(img_paths), self.img_size, self.img_size, 3), dtype=np.uint8)
		identities = np.empty(shape=(len(img_paths), ), dtype=np.int16)

		for i in tqdm(range(len(img_paths))):
			img = PIL.Image.open(img_paths[i])
			img = np.array(img)

			img = img[
				(img.shape[0] // 2 - self.crop_size // 2):(img.shape[0] // 2 + self.crop_size // 2),
				(img.shape[1] // 2 - self.crop_size // 2):(img.shape[1] // 2 + self.crop_size // 2)
			]

			img = PIL.Image.fromarray(img).resize(size=(self.img_size, self.img_size), resample=PIL.Image.BICUBIC)
			imgs[i] = np.array(img)

			identities[i] = unique_identities.index(identity_ids[i])

		return {
			'imgs': imgs,
			'identity': identities,
			'identity-ids': unique_identities
		}


supported_datasets = {
	'afhq': AFHQ,
	'celebahq': CelebAHQ,
	'ffhq': FFHQ,
	'celeba': CelebA
}
