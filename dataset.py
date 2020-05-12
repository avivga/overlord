import os
from abc import ABC, abstractmethod

import numpy as np


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


supported_datasets = {
	'cars3d': Cars3D
}
