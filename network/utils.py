import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class ImageTensorDataset(Dataset):

	def __init__(self, named_tensors):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors

	def __getitem__(self, index):
		item = {name: tensor[index] for name, tensor in self.named_tensors.items()}

		if 'img' in item:
			item['img'] = item['img'].float() / 255.0

		return item

	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)


class AugmentedDataset(ImageTensorDataset):

	def __init__(self, named_tensors, augmentation):
		super().__init__(named_tensors)

		transform_sequence = [
			transforms.ToPILImage(),

			transforms.RandomResizedCrop(
				size=tuple(self.named_tensors['img'].shape[2:4]),
				scale=augmentation['scale'],
				ratio=[1.0, 1.0]
			),

			transforms.RandomRotation(degrees=augmentation['rotation'])
		]

		if augmentation['flip_horizontal']:
			transform_sequence.append(transforms.RandomHorizontalFlip())

		transform_sequence.append(transforms.ToTensor())
		self.__transform = transforms.Compose(transform_sequence)

	def __getitem__(self, index):
		item = super().__getitem__(index)

		if 'mask' in item:
			item['img_masked'] = item['img'] * item['mask']
			img_for_augmentation = item['img_masked']
		else:
			img_for_augmentation = item['img']

		if isinstance(index, int):
			item['img_augmented'] = self.__transform(img_for_augmentation)
		else:
			item['img_augmented'] = torch.zeros_like(item['img'])
			for i in range(item['img'].shape[0]):
				item['img_augmented'][i] = self.__transform(img_for_augmentation[i])

		return item
