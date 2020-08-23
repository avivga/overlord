import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class NamedTensorDataset(Dataset):

	def __init__(self, named_tensors):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors

	def __getitem__(self, index):
		return {name: tensor[index] for name, tensor in self.named_tensors.items()}

	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)


class AugmentedDataset(NamedTensorDataset):

	def __init__(self, named_tensors):
		super().__init__(named_tensors)

		self.__transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomResizedCrop(tuple(self.named_tensors['img'].shape[2:4]), scale=[0.6, 1.0], ratio=[1.0, 1.0]),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		])

	def __getitem__(self, index):
		item = super().__getitem__(index)

		if isinstance(index, int):
			item['img_augmented'] = self.__transform(item['img'])
		else:
			item['img_augmented'] = torch.zeros_like(item['img'])
			for i in range(item['img'].shape[0]):
				item['img_augmented'][i] = self.__transform(item['img'][i])

		return item
