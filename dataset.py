import os
import argparse
from abc import ABC, abstractmethod
import glob

import imageio
import numpy as np
import scipy.io
import cv2

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
		class_ids = os.listdir(os.path.join(self._base_dir, self.split))

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
			'img': np.concatenate(imgs, axis=0),
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

		self.categories = [
			'Ani Redstart', 'Geococcyx', 'Pipit',
			'Albatross', 'Auklet Guillemot', 'Ani Blackbird Grackle Starling',
			'Bunting', 'Catbird Chat', 'Cormorant', 'Cowbird', 'Crow', 'Cuckoo',
			'Finch', 'Flycatcher Pewee Sayornis', 'Frigatebird', 'Fulmar', 'Goldfinch',
			'Grebe', 'Grosbeak Cardinal', 'Gull Kittiwake', 'Hummingbird', 'Jaeger', 'Jay',
			'Kingbird Kingfisher', 'Mallard Gadwall Loon', 'Merganser', 'Oriole Meadowlark',
			'Pelican', 'Puffin', 'Raven', 'Shrike', 'Sparrow Bobolink Junco Lark', 'Swallow',
			'Tanager', 'Tern', 'Towhee', 'Thrasher Mockingbird', 'Vireo', 'Violetear',
			'Warbler Widow Creeper Ovenbird Will', 'Waterthrush', 'Waxwing', 'Woodpecker Flicker Nuthatch',
			'Wren', 'Yellowthroat', 'Nighthawk', 'Nutcracker',
		]

	def find_category(self, img_name):
		for i, class_keys in enumerate(self.categories):
			for k in class_keys.split():
				if k in img_name.split('_'):
					return i

		raise Exception('no associated class')

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

			categories.append(self.find_category(img_name))

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

		args = parser.parse_args(extras)
		self.__dict__.update(vars(args))

	def read(self):
		img_paths = glob.glob(os.path.join(self._base_dir, self.split, '*.jpg'))

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


supported_datasets = {
	'afhq': AFHQ,
	'animalfaces': AnimalFaces,
	'cub': Cub,
	'edges2shoes': Edges2Shoes
}
