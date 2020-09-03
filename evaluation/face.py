import argparse
import imageio
import os
import json

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils import data
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import face_alignment
from facenet_pytorch import MTCNN, InceptionResnetV1

import hopenet
import utils


class NamedDataset(data.Dataset):

	def __init__(self, root, transform=None):
		self.paths = [os.path.join(root, f) for f in os.listdir(root)]
		self.transform = transform

	def __getitem__(self, index):
		path = self.paths[index]

		img = imageio.imread(path)
		if self.transform is not None:
			img = self.transform(img)

		return path, img

	def __len__(self):
		return len(self.paths)


@torch.no_grad()
def face_embeddings(img_dir):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	mtcnn = MTCNN(image_size=128, margin=0, device=device)
	facenet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

	embeddings = {}
	paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
	for path in tqdm(paths):
		img = Image.open(path)
		img_cropped = mtcnn(img)
		if img_cropped is None:
			print('failed to find face in {}'.format(path))
			continue

		embedding = facenet(img_cropped.unsqueeze(0).to(device))
		embeddings[path] = embedding.squeeze(0).cpu().numpy()

	return embeddings


@torch.no_grad()
def head_poses(img_dir, hopenet_path):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
	model.load_state_dict(torch.load(hopenet_path))
	model.to(device)
	model.eval()

	transformations = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	idx_tensor = torch.FloatTensor(np.arange(66)).to(device)
	dataset = NamedDataset(root=img_dir, transform=transformations)

	data_loader = DataLoader(
		dataset, batch_size=64,
		shuffle=False, pin_memory=True, drop_last=False
	)

	head_poses = {}
	for batch_paths, batch_imgs in tqdm(data_loader):
		yaw, pitch, roll = model(batch_imgs.to(device))

		yaw_predicted = utils.softmax_temperature(yaw.data, 1)
		pitch_predicted = utils.softmax_temperature(pitch.data, 1)
		roll_predicted = utils.softmax_temperature(roll.data, 1)

		yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu().numpy() * 3 - 99
		pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu().numpy() * 3 - 99
		roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu().numpy() * 3 - 99

		for i, path in enumerate(batch_paths):
			head_poses[path] = {
				'yaw': yaw_predicted[i],
				'pitch': pitch_predicted[i],
				'roll': roll_predicted[i]
			}

	return head_poses


def map_by_id(d):
	return {
		os.path.splitext(os.path.basename(path))[0]: v
		for path, v in d.items()
	}


@torch.no_grad()
def eval_metrics(args):
	translations_dir = os.path.join(args.eval_dir, 'translations')

	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	content_landmarks = map_by_id(fa.get_landmarks_from_directory(path=os.path.join(translations_dir, 'content')))
	translation_landmarks = map_by_id(fa.get_landmarks_from_directory(path=os.path.join(translations_dir, 'translation')))

	style_embeddings = map_by_id(face_embeddings(img_dir=os.path.join(translations_dir, 'style')))
	translation_embeddings = map_by_id(face_embeddings(img_dir=os.path.join(translations_dir, 'translation')))

	head_poses_content = map_by_id(head_poses(img_dir=os.path.join(translations_dir, 'content'), hopenet_path=args.hopenet_path))
	head_poses_translation = map_by_id(head_poses(img_dir=os.path.join(translations_dir, 'translation'), hopenet_path=args.hopenet_path))

	losses = {
		'landmarks': [],
		'embeddings': [],
		'head_pose': {
			'yaw': [],
			'pitch': [],
			'roll': []
		}
	}

	for translation_id in translation_landmarks.keys():
		content_id, style_id = translation_id.split('-')

		if style_id not in style_embeddings or translation_id not in translation_embeddings:
			continue

		losses['landmarks'].append(
			np.mean(np.abs(content_landmarks[content_id][0] - translation_landmarks[translation_id][0]))
		)

		losses['embeddings'].append(
			np.mean(np.abs(style_embeddings[style_id] - translation_embeddings[translation_id]))
		)

		losses['head_pose']['yaw'].append(
			np.mean(np.abs(head_poses_content[content_id]['yaw'] - head_poses_translation[translation_id]['yaw']))
		)

		losses['head_pose']['pitch'].append(
			np.mean(np.abs(head_poses_content[content_id]['pitch'] - head_poses_translation[translation_id]['pitch']))
		)

		losses['head_pose']['roll'].append(
			np.mean(np.abs(head_poses_content[content_id]['roll'] - head_poses_translation[translation_id]['roll']))
		)

	summary = {
		'landmarks': np.mean(losses['landmarks']).item(),
		'embeddings': np.mean(losses['embeddings']).item(),
		'head_pose': {
			'yaw': np.mean(losses['head_pose']['yaw']).item(),
			'pitch': np.mean(losses['head_pose']['pitch']).item(),
			'roll': np.mean(losses['head_pose']['roll']).item()
		}
	}

	with open(os.path.join(args.eval_dir, 'face.json'), 'w') as f:
		json.dump(summary, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval-dir', type=str, required=True)
	parser.add_argument('--hopenet-path', type=str, required=True)
	args = parser.parse_args()

	eval_metrics(args)
