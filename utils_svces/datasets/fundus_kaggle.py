from __future__ import print_function, division
import os
from typing import Tuple, List, Dict

import torch
import sys
from torchvision import datasets
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder, has_file_allowed_extension
from .balanced_batch_sampler import BalancedBatchSampler, ScheduledWeightedSampler
from .paths import get_fundusKaggle_path
from torch.distributions.uniform import Uniform

# channel means and standard deviations of kaggle dataset, computed by origin author
MEAN = torch.tensor([0.4913997551666284, 0.48215855929893703,
					 0.4465309133731618])  # [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = torch.tensor([0.24703225141799082, 0.24348516474564,
					0.26158783926049628])  # [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]

# for color augmentation, computed by origin author
U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                  [-0.5989477, -0.02304967, -0.80036049],
                  [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# mean std
# scale to 0..1
pil_to_tensor = lambda img: transforms.ToTensor()(img).unsqueeze_(0)

# don't leave 0..255 before
tensor_to_pil = lambda tensor: transforms.ToPILImage()(tensor.squeeze_(0))

class kaggleFundusDataset(DatasetFolder):

	def __init__(self, csv_file, data_dir, csv_dir, transform=None, binary=True, get_name=False):
		super(DatasetFolder, self).__init__(root=data_dir, transform=transform)

		self.fundus_df = pd.read_csv(os.path.join(csv_dir, csv_file),
								 low_memory=False)
		self.data_dir = data_dir
		extensions = ('.jpeg', )
		classes, class_to_idx = self._find_classes()
		samples = self._make_dataset(class_to_idx, extensions=extensions,
									 is_valid_file=has_file_allowed_extension)
		if len(samples) == 0:
			raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
																				 "Supported extensions are: " + ",".join(
				extensions)))


		self.transform = transform
		self.radius = 224/10
		# Use 'dr' for binary labels and 'level' for 5-class classification
		self.labels_type = 'dr' if binary else 'level'
		self.get_name = get_name
		self.classes = classes
		self.class_to_idx = class_to_idx
		self.samples = samples
		self.targets = [s[1] for s in samples]

	def _find_classes(self, dir: str=None) -> Tuple[List[str], Dict[str, int]]:
		classes = self.fundus_df.dr.tolist()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

	def _make_dataset(self, class_to_idx, extensions=None, is_valid_file=None) -> List[Tuple[str, int]]:
		images = []
		for target_id in sorted(class_to_idx.values()):
			target = self.fundus_df.filename[target_id]
			path = os.path.join(self.data_dir, target)

			if is_valid_file(path, extensions):
				item = (path, target_id)
				images.append(item)
		return images

	def __len__(self):
		return len(self.fundus_df)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = os.path.join(self.data_dir, self.fundus_df['filename'].iloc[idx])
		label = self.fundus_df[self.labels_type].iloc[idx]
		name = self.fundus_df['image'].iloc[idx]

		with Image.open(img_name) as img:
			if self.transform is not None:
				img = self.transform(img)

		if self.get_name:
			return img, int(label), img_name
		else:
			return img, int(label)


def get_FundusKaggle(split='train', batch_size=16, shuffle=None, size=224, num_workers=2, binary=True,
					 augm_type=None, get_name=False, balanced=False,
					 preprocess=None, bg_kernel='median', clahe=True,
					 project_folder=None, data_folder=None, version='v2'):
	# ToDo: make the same dir for both
	print(split, batch_size, shuffle, size, num_workers, binary, augm_type, get_name, balanced)
	print('clahe', clahe, preprocess)
	#data_dir = get_fundusKaggle_path(data_folder, background_subtraction=background_subtraction)
	data_dir = get_fundusKaggle_path(data_folder, background_subtraction=False, version=version,
									 clahe=clahe, preprocess=preprocess)
	# ToDO: move to the dataset dir
	csv_dir = os.path.join(project_folder, 'datasets')

	if split in ['train', 'test', 'val']:
		version = 'v3'
		version_file = '_'+version if version is not None else ''
		csv_file = f'kaggle_gradable_{split}{version_file}.csv'
		print('csv file', csv_file)
	else:
		raise ValueError(f'There is no split {split}!')

	#elif partition == 'test':
	#	csv_file = 'kaggle_gradable_val.csv'
	#else:
	#	csv_file = None
	#	raise ValueError('Invalid value for partition. Partition can take values train, val or test')

	# Should we leave this transform?
	"""
	transform_train = transforms.Compose([transforms.Resize(size),
					      transforms.RandomCrop(size, padding=4),
					      transforms.RandomHorizontalFlip(),
					      transforms.ToTensor(),
					      ])


	transform_test = transforms.Compose([transforms.Resize(size),
					     transforms.ToTensor(),
					     ])
	"""

	transform_test = transforms.Compose([transforms.Resize(size),
										 transforms.ToTensor(),
										 ])

	if augm_type.lower() == 'murat':
		transform_train = transforms.Compose([transforms.Resize(size),
											  transforms.RandomApply([
													  transforms.RandomCrop(size, padding=int(0.15 * size)),  # 0.15
												  # RandomBrightnessCustom(0.15),
												  #RandomContrastCustom((0.5, 2.5)),
												  transforms.ColorJitter(brightness=(0.5, 0.9), contrast=(0.5, 2.5),
																		 saturation=(0.5, 1.5), hue=(-0.15, 0.15)),
												  transforms.RandomHorizontalFlip(),
												  transforms.RandomVerticalFlip(),
												  transforms.RandomAffine(degrees=(-15, 15),
																		  translate=(0.048, 0.048),
																		  )
												], p=0.9),
											  transforms.ToTensor()
											  ])
	elif augm_type.lower() == 'o_o':
		transform_train = transforms.Compose([transforms.RandomResizedCrop(
													  size=size,
													  scale=(1 / 1.15, 1.15),
													  ratio=(0.7561, 1.3225)
												  ),
												transforms.RandomAffine(
													degrees=(-180, 180),
													translate=(40 / 224, 40 / 224),
													scale=None,
													shear=None
												),
												transforms.RandomHorizontalFlip(),
												transforms.RandomVerticalFlip(),
												transforms.ToTensor(),
												transforms.Normalize(tuple(MEAN), tuple(STD)),
												KrizhevskyColorAugmentation(sigma=0.5)
											])

		transform_test = transforms.Compose([transforms.Resize(size),
											 transforms.ToTensor(),
											 transforms.Normalize(tuple(MEAN), tuple(STD))]
											)
	elif augm_type.lower() == 'none':
		transform_train = transforms.Compose([transforms.Resize(size),
										     transforms.ToTensor()
										     ])
	else:
		raise ValueError(f'Transform {augm_type} is not implemented')



	transform = transform_train if split=='train' else transform_test

	if shuffle is None:
		if split=='train':
			shuffle = True
		else:
			shuffle = False
	# kaggleFundusDataset
	dataset = kaggleFundusDataset(csv_file, data_dir, csv_dir, preprocess=preprocess, bg_kernel=bg_kernel,
								  clahe=clahe, transform=transform, binary=binary,
								  get_name=get_name)

	if balanced and split == 'train':
		# Is it correct?
		if augm_type.lower() == 'o_o':
			train_targets = dataset.classes
			sampler = ScheduledWeightedSampler(len(dataset), train_targets, replacement=True)
		else:
			sampler = BalancedBatchSampler(dataset)
		shuffle = False
	else:
		sampler = None
	dataloader = DataLoader(dataset, batch_size=batch_size,
				shuffle=shuffle, num_workers=num_workers,
							sampler=sampler)

	return dataloader


class RandomBrightnessCustom(torch.nn.Module):
	"""RandomContrast implementation of tf
	Args:
		factor_bounds (tuple): as in tf
	"""

	def __init__(self, delta):
		super().__init__()
		self.delta = delta

	def forward(self, image):
		"""
		Args:
			img (PIL Image or Tensor): Image to be brightened.

		Returns:
			PIL Image or Tensor: Randomly brightened image.
		"""
		image = pil_to_tensor(image)
		# Generates uniformly distributed random samples from the half-open interval [low, high)
		delta = Uniform(-self.delta, self.delta).sample()
		# Add the delta to each pixel of the image
		adjusted_image = image + delta
		return tensor_to_pil(adjusted_image)

	def __repr__(self):
		return self.__class__.__name__ + '(factor_bounds={})'.format(self.factor_bounds)


class RandomContrastCustom(torch.nn.Module):
	"""RandomContrast implementation of tf
	Args:
		factor_bounds (tuple): as in tf
	"""

	def __init__(self, factor_bounds):
		super().__init__()
		self.factor_bounds = factor_bounds

	def forward(self, image):
		"""
		Args:
			img (PIL Image or Tensor): Image to be flipped.

		Returns:
			PIL Image or Tensor: Randomly contrasted image.
		"""
		image = pil_to_tensor(image)
		factor = Uniform(self.factor_bounds[0], self.factor_bounds[1]).sample()
		mean = image.mean([1, 2])
		# Add the delta to each pixel of the image
		adjusted_image = (image - mean)*factor + mean
		return tensor_to_pil(adjusted_image)

	def __repr__(self):
		return self.__class__.__name__ + '(factor_bounds={})'.format(self.factor_bounds)


class KrizhevskyColorAugmentation(object):
	def __init__(self, sigma=0.5):
		self.sigma = sigma
		self.mean = torch.tensor([0.0])
		self.deviation = torch.tensor([sigma])

	def __call__(self, img, color_vec=None):
		sigma = self.sigma
		if color_vec is None:
			if not sigma > 0.0:
				color_vec = torch.zeros(3, dtype=torch.float32)
			else:
				color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
			color_vec = color_vec.squeeze()

		alpha = color_vec * EV
		noise = torch.matmul(U, alpha.t())
		noise = noise.view((3, 1, 1))
		return img + noise

	def __repr__(self):
		return self.__class__.__name__ + '(sigma={})'.format(self.sigma)
