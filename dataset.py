import os
import argparse
from abc import ABC, abstractmethod

import imageio
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm

import face_alignment


class DataSet(ABC):

	def __init__(self, base_dir=None, extras=None):
		super().__init__()
		self._base_dir = base_dir
		self._extras = extras

	@abstractmethod
	def read(self):
		pass


class Cub(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir)

		parser = argparse.ArgumentParser()
		parser.add_argument('-sp', '--split', type=str, choices=['train', 'val'], required=True)
		parser.add_argument('-is', '--img-size', type=int, default=256)

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
		keypoints = []

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

			img_scale = self.img_size / box_length
			imgs.append(cv2.resize(img_cropped, dsize=(self.img_size, self.img_size)))

			img_keypoints = img_struct.parts.T.astype(np.float32)
			visible = (img_keypoints[:, 2] > 0)
			img_keypoints[visible, :2] = img_keypoints[visible, :2] - 1

			img_keypoints[visible, 0] = (img_keypoints[visible, 0] - x1) * img_scale
			img_keypoints[visible, 0] = 2.0 * (img_keypoints[visible, 0] / self.img_size) - 1

			img_keypoints[visible, 1] = (img_keypoints[visible, 1] - y1) * img_scale
			img_keypoints[visible, 1] = 2.0 * (img_keypoints[visible, 1] / self.img_size) - 1

			keypoints.append(img_keypoints.reshape((-1, )))
			categories.append(self.find_category(img_name))

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.array(categories),
			'content': np.stack(keypoints, axis=0)
		}


class Pascal3D(DataSet):

	class Pascal(object):

		def __init__(self, directory, class_ids, set_name):
			self.name = 'pascal'
			self.directory = directory
			self.class_ids = class_ids
			self.set_name = set_name
			self.image_size = 224

			self.images_original = {}
			self.images_ref = {}
			self.bounding_boxes = {}
			self.rotation_matrices = {}
			self.voxels = {}
			self.num_data = {}

			for class_id in class_ids:
				data = np.load(os.path.join(directory, '%s_%s.npz' % (class_id, set_name)), allow_pickle=True, encoding='latin1')
				self.images_original[class_id] = data['images']
				self.images_ref[class_id] = data['images_ref']
				self.bounding_boxes[class_id] = data['bounding_boxes']
				self.rotation_matrices[class_id] = data['rotation_matrices']
				self.voxels[class_id] = data['voxels']

				if set_name == 'train':
					# add ImageNet data
					data = np.load(os.path.join(directory, '%s_%s.npz' % (class_id, 'imagenet')), allow_pickle=True,  encoding='latin1')
					self.images_original[class_id] = np.concatenate(
						(self.images_original[class_id], data['images']), axis=0)
					self.images_ref[class_id] = np.concatenate(
						(self.images_ref[class_id], data['images_ref']), axis=0)
					self.bounding_boxes[class_id] = np.concatenate(
						(self.bounding_boxes[class_id], data['bounding_boxes']), axis=0)
					self.rotation_matrices[class_id] = np.concatenate(
						(self.rotation_matrices[class_id], data['rotation_matrices']), axis=0)
					self.voxels[class_id] = np.concatenate((self.voxels[class_id], data['voxels']), axis=0)

				self.images_ref[class_id] = self.images_ref[class_id].transpose((0, 3, 1, 2))
				self.num_data[class_id] = self.images_ref[class_id].shape[0]

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
		pascal = Pascal3D.Pascal(directory=self._base_dir, class_ids=[self.category], set_name=self.split)

		imgs = pascal.images_ref[self.category].transpose(0, 2, 3, 1)[..., :3]
		masks = pascal.images_ref[self.category].transpose(0, 2, 3, 1)[..., 3]

		imgs[masks < 50] = 255
		imgs = [cv2.resize(imgs[i], dsize=(self.img_size, self.img_size)) for i in range(imgs.shape[0])]

		return {
			'img': np.stack(imgs, axis=0),
			'class': np.load(self.classes_path)['classes']
		}


class CelebA(DataSet):

	def __init__(self, base_dir, extras):
		super().__init__(base_dir, extras)

		parser = argparse.ArgumentParser()
		parser.add_argument('-cs', '--crop-size', type=int, nargs=2, default=(128, 128))
		parser.add_argument('-ts', '--target-size', type=int, nargs=2, default=(128, 128))
		parser.add_argument('-ni', '--n-images', type=int, required=False)

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

		if self.n_images:
			idx = np.random.choice(len(img_paths), size=self.n_images)
			return np.array(img_paths)[idx].tolist(), np.array(identities)[idx].tolist()

		else:
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

		imgs = np.empty(shape=(len(img_paths), self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
		identities = np.empty(shape=(len(img_paths), ), dtype=np.int32)
		attributes = np.empty(shape=(len(img_paths), 40), dtype=np.int8)
		landmarks = np.zeros(shape=(len(img_paths), 68*2), dtype=np.int16)

		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

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

			preds = fa.get_landmarks(imgs[i])
			if preds is None or len(preds) == 0:
				continue

			landmarks[i] = preds[0].flatten().astype(np.int16)

		return {
			'img': imgs,
			'class': identities,
			'attributes': attributes,
			'landmarks': landmarks
		}


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


supported_datasets = {
	'afhq': AFHQ,
	'cub': Cub,
	'pascal3d': Pascal3D,
	'celeba': CelebA
}
