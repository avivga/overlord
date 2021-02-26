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
		parser.add_argument('-is', '--img-size', type=int, default=128)

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
		parser.add_argument('-pr', '--part', type=str, required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)

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

		mask_paths = glob.glob(os.path.join(self._base_dir, 'CelebAMask-HQ', 'CelebAMask-HQ-mask-anno', '*', '*_{}.png'.format(self.part)))
		mask_paths = {os.path.basename(p).split('_')[0]: p for p in mask_paths}

		imgs = np.empty(shape=(len(img_names), self.img_size, self.img_size, 3), dtype=np.uint8)
		masks = np.empty(shape=(len(img_names), self.img_size, self.img_size), dtype=np.uint8)
		attributes = np.full(shape=(len(img_names), 40), fill_value=-1, dtype=np.int16)

		for i, img_name in enumerate(tqdm(img_names)):
			img_path = os.path.join(self._base_dir, 'x1024', img_name)
			img = imageio.imread(img_path)
			imgs[i] = cv2.resize(img, dsize=(self.img_size, self.img_size))

			img_id = os.path.splitext(img_name)[0]
			mask_id = '{:05d}'.format(int(img_id))
			if mask_id in mask_paths:
				mask = imageio.imread(mask_paths[mask_id])[..., 0]
				mask = mask // 255
				masks[i] = cv2.resize(mask, dsize=(self.img_size, self.img_size))
			else:
				masks[i] = np.zeros(shape=(self.img_size, self.img_size), dtype=np.uint8)

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
		parser.add_argument('-is', '--img-size', type=int, default=128)

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

		return {
			'img': imgs,
			'age': age,
			'gender': gender
		}


class Carnivores(DataSet):  # from imagenet

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-cn', '--class-name-list', type=str, required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)
		parser.add_argument('-nc', '--n-classes', type=int, required=False)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		with open(self.class_name_list, 'r') as fp:
			class_ids = fp.read().splitlines()

		if self.n_classes:
			class_ids = random.sample(class_ids, k=self.n_classes)

		imgs = []
		classes = []

		for class_idx, class_id in enumerate(class_ids):
			img_names = os.listdir(os.path.join(self._base_dir, 'imgs', self.split, class_id))

			pbar = tqdm(img_names)
			for img_name in pbar:
				pbar.set_description_str('preprocessing {}[{}]'.format(class_id, img_name))
				img = imageio.imread(os.path.join(self._base_dir, 'imgs', self.split, class_id, img_name))
				if len(img.shape) == 2:
					img = np.tile(img[..., np.newaxis], reps=(1, 1, 3))

				img_id = os.path.splitext(img_name)[0]
				annotation_path = os.path.join(self._base_dir, 'annotations', self.split, class_id, img_id + '.xml')
				if not os.path.exists(annotation_path):
					continue

				root = ElementTree.parse(annotation_path).getroot()
				bounding_boxes = root.findall('object/bndbox')

				# bb = bounding_boxes[0]  # rest are sometimes broken
				for bb in bounding_boxes:
					x_min, x_max, y_min, y_max = (
						int(bb.find('xmin').text), int(bb.find('xmax').text),
						int(bb.find('ymin').text), int(bb.find('ymax').text)
					)

					bb_width = x_max - x_min
					bb_height = y_max - y_min

					size = max(bb_width, bb_height)
					if size < 64:
						continue

					x_padding = (size - bb_width) // 2
					y_padding = (size - bb_height) // 2

					x_min = max(x_min - x_padding, 0)
					y_min = max(y_min - y_padding, 0)

					x_max = x_min + size
					y_max = y_min + size

					if x_max >= img.shape[1] or y_max >= img.shape[0]:
						img_padded = np.pad(img, ((0, max(y_max - img.shape[0], 0)), (0, max(x_max - img.shape[1], 0)), (0, 0)), mode='reflect')
					else:
						img_padded = img

					img_cropped = img_padded[y_min:y_max, x_min:x_max]
					img_resized = cv2.resize(img_cropped, dsize=(self.img_size, self.img_size))

					imgs.append(img_resized)
					classes.append(class_idx)

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.array(classes, dtype=np.int32)
		}


class Horse2Zebra(DataSet):  # from imagenet

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)
		parser.add_argument('-ad', '--annotation-dir', type=str, required=True)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		imgs = []
		classes = []

		class_map = {
			'A': 'n02381460',
			'B': 'n02391049'
		}

		for class_idx, class_id in enumerate(['A', 'B']):
			img_names = os.listdir(os.path.join(self._base_dir, '{}{}'.format(self.split, class_id)))

			pbar = tqdm(img_names)
			for img_name in pbar:
				pbar.set_description_str('preprocessing {}[{}]'.format(class_id, img_name))
				img = imageio.imread(os.path.join(self._base_dir, '{}{}'.format(self.split, class_id), img_name))
				if len(img.shape) == 2:
					img = np.tile(img[..., np.newaxis], reps=(1, 1, 3))

				img_id = os.path.splitext(img_name)[0]
				annotation_path = os.path.join(self.annotation_dir, class_map[class_id], img_id + '.xml')
				if not os.path.exists(annotation_path):
					img_resized = cv2.resize(img, dsize=(self.img_size, self.img_size))
					imgs.append(img_resized)
					classes.append(class_idx)
					continue

				root = ElementTree.parse(annotation_path).getroot()
				bounding_boxes = root.findall('object/bndbox')

				# bb = bounding_boxes[0]  # rest are sometimes broken
				for bb in bounding_boxes:
					x_min, x_max, y_min, y_max = (
						int(bb.find('xmin').text), int(bb.find('xmax').text),
						int(bb.find('ymin').text), int(bb.find('ymax').text)
					)

					bb_width = x_max - x_min
					bb_height = y_max - y_min

					size = max(bb_width, bb_height)
					# if size < 64:
					# 	continue

					x_padding = (size - bb_width) // 2
					y_padding = (size - bb_height) // 2

					x_min = max(x_min - x_padding, 0)
					y_min = max(y_min - y_padding, 0)

					x_max = x_min + size
					y_max = y_min + size

					if x_max >= img.shape[1] or y_max >= img.shape[0]:
						img_padded = np.pad(img, ((0, max(y_max - img.shape[0], 0)), (0, max(x_max - img.shape[1], 0)), (0, 0)), mode='reflect')
					else:
						img_padded = img

					img_cropped = img_padded[y_min:y_max, x_min:x_max]
					img_resized = cv2.resize(img_cropped, dsize=(self.img_size, self.img_size))

					imgs.append(img_resized)
					classes.append(class_idx)

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.array(classes, dtype=np.int32)
		}


class AnimalFaces(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sl', '--split-list', type=str, required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)
		parser.add_argument('-ss', '--shortest-size', type=int, default=140)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		imgs = []
		classes = []

		with open(self.split_list, 'r') as fp:
			paths = fp.read().splitlines()

		for path in tqdm(paths):
			img = imageio.imread(os.path.join(self._base_dir, path))

			shortest = min(img.shape[0], img.shape[1])
			factor = self.shortest_size / shortest

			img = cv2.resize(img, dsize=(int(img.shape[1] * factor), int(img.shape[0] * factor)))

			top = np.random.choice(np.arange(img.shape[0] - self.img_size + 1))
			left = np.random.choice(np.arange(img.shape[1] - self.img_size + 1))

			img_cropped = img[top:top+self.img_size, left:left+self.img_size]
			imgs.append(img_cropped)

			class_id, img_name = path.split('/')
			classes.append(class_id)

		unique_class_ids = list(set(classes))
		classes = [unique_class_ids.index(c) for c in classes]

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.array(classes)
		}


class Cub(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		data = scipy.io.loadmat(os.path.join(self._base_dir, 'from_cmr', 'data', '{}_cub_cleaned.mat'.format(self.split)), struct_as_record=False, squeeze_me=True)

		n_images = data['images'].shape[0]
		imgs = []
		categories = []

		for i in range(n_images):
			img_struct = data['images'][i]
			img_name = os.path.basename(img_struct.rel_path)
			img_path = os.path.join(self._base_dir, 'images', img_struct.rel_path)

			img = imageio.imread(img_path)
			if len(img.shape) == 2:
				img = np.tile(img[..., np.newaxis], reps=(1, 1, 3))

			img[img_struct.mask == 0] = 255

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

			img_cropped = img[y1:y2, x1:x2]
			imgs.append(cv2.resize(img_cropped, dsize=(self.img_size, self.img_size)))

			category_id = img_struct.rel_path.split('/')[0]
			categories.append(category_id)

		unique_categories = sorted(list(set(categories)))
		categories = list(map(lambda c: unique_categories.index(c), categories))

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.array(categories)
		}


class Edges2Shoes(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)
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


class Cars3D(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir, extras)

		self.__data_path = os.path.join(base_dir, 'cars3d.npz')

	def read(self):
		imgs = np.load(self.__data_path)['imgs']

		elevations = np.empty(shape=(imgs.shape[0],), dtype=np.int64)
		azimuths = np.empty(shape=(imgs.shape[0], ), dtype=np.int64)
		objects = np.empty(shape=(imgs.shape[0],), dtype=np.int64)

		for elevation_id in range(4):
			for azimuth_id in range(24):
				for object_id in range(183):
					img_idx = elevation_id * 24 * 183 + azimuth_id * 183 + object_id

					elevations[img_idx] = elevation_id
					azimuths[img_idx] = azimuth_id
					objects[img_idx] = object_id

		return {
			'img': imgs,
			'elevation': elevations,
			'azimuth': azimuths,
			'class': objects
		}


class AB(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'test'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=128)

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		img_paths_a = glob.glob(os.path.join(self._base_dir, self.split + 'A', '*.jpg'))
		img_paths_b = glob.glob(os.path.join(self._base_dir, self.split + 'B', '*.jpg'))

		imgs_a = np.empty(shape=(len(img_paths_a), self.img_size, self.img_size, 3), dtype=np.uint8)
		imgs_b = np.empty(shape=(len(img_paths_b), self.img_size, self.img_size, 3), dtype=np.uint8)

		for i in range(len(img_paths_a)):
			img = imageio.imread(img_paths_a[i])

			if len(img.shape) == 2:
				img = np.tile(img[..., np.newaxis], reps=(1, 1, 3))

			imgs_a[i] = cv2.resize(img, dsize=(self.img_size, self.img_size))

		for i in range(len(img_paths_b)):
			img = imageio.imread(img_paths_b[i])

			if len(img.shape) == 2:
				img = np.tile(img[..., np.newaxis], reps=(1, 1, 3))

			imgs_b[i] = cv2.resize(img, dsize=(self.img_size, self.img_size))

		imgs = np.concatenate((imgs_a, imgs_b), axis=0)
		classes = np.concatenate((
			np.zeros(shape=(imgs_a.shape[0], ), dtype=np.uint8),
			np.ones(shape=(imgs_b.shape[0], ), dtype=np.uint8)
		), axis=0)

		return {
			'img': imgs,
			'class': classes
		}


supported_datasets = {
	'afhq': AFHQ,
	'celebahq': CelebAHQ,
	'ffhq': FFHQ,
	'animalfaces': AnimalFaces,
	'carnivores': Carnivores,
	'horse2zebra': Horse2Zebra,
	'cub': Cub,
	'edges2shoes': Edges2Shoes,
	'celeba': CelebA,
	'cars3d': Cars3D,
	'ab': AB
}
