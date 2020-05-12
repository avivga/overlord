from torch.utils.data.dataset import Dataset


class AverageMeter:

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class NamedTensorDataset(Dataset):

	def __init__(self, named_tensors):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors

	def __getitem__(self, index):
		return {name: tensor[index] for name, tensor in self.named_tensors.items()}

	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)

	def subset(self, indices):
		return NamedTensorDataset(self[indices])
