import math
import socket

import torch
import torch.distributions
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler, Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from .paths import get_imagenet_path
from .imagenet_augmentation import get_imageNet_augmentation
from PIL import Image
import pickle
import numpy as np


def get_base_dir(project_folder=None):
    if project_folder is not None:
        return project_folder


DEFAULT_TRAIN_BATCHSIZE = 128
DEFAULT_TEST_BATCHSIZE = 128

def get_max_batch_dataset(dataset, max_batch):
    # Split the indices in a stratified way
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=max_batch)  # , stratify=dataset.targets)

    # Warp into Subsets and DataLoaders
    dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return dataset, test_dataset

def convex_alpha_combination(img, watermark_rgba, intensity=1.0):
    alpha_mask = intensity * watermark_rgba[-1,:,:][None,:,:]
    cvx_combination = alpha_mask * watermark_rgba[0:3, :, :] + (1.0 - alpha_mask) * img[:,:,:]
    return cvx_combination

class Imagenet_Watermark(datasets.ImageNet):
    def __init__(self, root, split='train', watermark_type='hicker', apply_pre_transform=True, download=None,
                 intensity=0.8, color=None, project_folder=None, **kwargs):
        super(Imagenet_Watermark, self).__init__(root, split, download, **kwargs)

        if watermark_type == 'hicker':
            print('using hicker watermark')
            hicker_file = f'{get_base_dir(project_folder)}/notebooks/Hicker_Min.png'
            to_tensor = transforms.ToTensor()
            mask = to_tensor(Image.open(hicker_file))

            #normalize so max value is 1
            mask = mask / torch.max(mask)
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
        elif watermark_type == 'istockphoto':
            file = f'{get_base_dir(project_folder)}/notebooks/Wm_binary_scaled.pt'
            mask = torch.load(file)
            mask = mask.repeat([3, 1, 1])
        else:
            raise NotImplementedError()

        #add alpha dimension
        mask_alpha = torch.zeros((4, mask.shape[1], mask.shape[2]))
        mask_alpha[0:3, :, :] = mask
        mask_alpha[-1, :, :] = mask[0, :, :]

        if color is not None:
            #color should be a tensor of size 3
            color_extended = color[:, None, None]
            mask_alpha[0:3] = mask[0:3] * color_extended

        self.mask = mask_alpha

        self.intensity = intensity
        self.apply_pre_transform = apply_pre_transform

        print('using intensity for scaled wm:', self.intensity)
        assert self.mask.min() >= 0 and self.mask.max() <= 1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.apply_pre_transform:
            resize = transforms.Resize((sample.height, sample.width))
            to_tensor = transforms.ToTensor()
            to_pil = transforms.ToPILImage()
            scaled_watermark = resize(self.mask)
            sample = convex_alpha_combination(to_tensor(sample), scaled_watermark, self.intensity)
            sample = to_pil(sample)

            if self.transform is not None:
                sample = self.transform(sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            resize = transforms.Resize((sample.shape[1], sample.shape[2]))
            scaled_watermark = resize(self.mask)
            sample = convex_alpha_combination(sample, scaled_watermark, self.intensity)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_ImageNet1000_idx(idx_path, batch_size=None, shuffle=False, no_subset_sampler=False,
                         wm_intensity=0., watermark_type='istockphoto', model_name=None, project_folder=None,
                         data_folder=None):
    if batch_size == None:
        batch_size = DEFAULT_TEST_BATCHSIZE


    if 'swin_large_patch4_window12_384' in model_name:
        print('get_ImageNet1000_idx: using transforms in dataloader for swin_large_patch4_window12_384')
        img_size = 384
        crop_pct = 1.0

        scale_size = int(math.floor(img_size / crop_pct))

        transform = transforms.Compose([
            transforms.Resize(scale_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    elif 'tf_efficientnet_b7' in model_name:
        print('get_ImageNet1000_idx: using transforms in dataloader for tf_efficientnet_b7')
        img_size = 600
        crop_pct = 0.949

        scale_size = int(math.floor(img_size / crop_pct))

        transform = transforms.Compose([
            transforms.Resize(scale_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    elif 'swin_large_patch4_window7_224' in model_name:
        print('get_ImageNet1000_idx: using transforms in dataloader for swin_large_patch4_window7_224')
        img_size = 224
        crop_pct = 0.9

        scale_size = int(math.floor(img_size / crop_pct))

        transform = transforms.Compose([
            transforms.Resize(scale_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    elif 'convnext_large_in22ft1k' in model_name:
        print('get_ImageNet1000_idx: using transforms in dataloader for convnext_large_in22ft1k')
        img_size = 224
        crop_pct = 0.875

        scale_size = int(math.floor(img_size / crop_pct))

        transform = transforms.Compose([
            transforms.Resize(scale_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    else:
        print('get_ImageNet1000_idx: using standard transforms in dataloader')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if no_subset_sampler:
        pass
    else:
        print('subset sampler from', idx_path)
        idx = np.load(idx_path)

    #train = False
    shuffle = False

    if watermark_type == 'istockphoto':
        print('using istockphoto')
        color = torch.FloatTensor([1., 1., 1.])
        dataset = Imagenet_Watermark(get_imagenet_path(data_folder), transform=transform, split='val', watermark_type='istockphoto',
                                     color=color, intensity=wm_intensity, apply_pre_transform=False, project_folder=project_folder)
    elif watermark_type == 'hicker':
        color = torch.FloatTensor([87, 91, 100]) / 255.
        dataset = Imagenet_Watermark(get_imagenet_path(data_folder), transform=transform, split='val', watermark_type='hicker',
                                  color=color, intensity=wm_intensity, apply_pre_transform=True, project_folder=project_folder)
    else:
        raise ValueError('Not implemented.')
    #dataset = datasets.ImageNet(path, split='val' if not train else 'train', transform=transform)
    print('dataset and batch size are', len(dataset), batch_size)
    if no_subset_sampler:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=8)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=8, sampler=SubsetSampler(idx))

    return loader

def get_restrictedImageNet(train=True, batch_size=None, shuffle=None, augm_type='test',
                           balanced=True, num_workers=8, size=224, config_dict=None, num_samples=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)

    if not train and augm_type != 'test' and augm_type != 'none':
        print('Warning: ImageNet test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_imagenet_path()

    if train == True:
        dataset = RestrictedImageNet(path, split='train', transform=transform, balanced=balanced)
    else:
        dataset = RestrictedImageNet(path, split='val', transform=transform, balanced=balanced)

    if num_samples is not None and not balanced:
        dataset, _ = get_max_batch_dataset(dataset, 200)
        print('restrImageNet reduced size is', len(dataset))

    if balanced:
        sampler = RestrictedImagenetBalancedSampler(dataset, shuffle)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=sampler, num_workers=num_workers)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'RestrictedImageNet'
        config_dict['Balanced'] = balanced
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader


def get_restrictedImageNetValidationTestSplit(val_split=True, batch_size=128, shuffle=False, augm_type='test', num_workers=8, size=224):
    try:
        idxs = np.loadtxt('idxs_rimgnet.txt', dtype=int)
    except:
        idxs = []

    if not val_split:
        all_idcs = np.arange(10150)
        idxs = np.setdiff1d(all_idcs, idxs) #test idcs

    if shuffle:
        sampler = SubsetRandomSampler(idxs)
    else:
        sampler = SubsetSampler(idxs)

    transform = get_imageNet_augmentation(type=augm_type, out_size=size)

    path = get_imagenet_path()
    dataset = RestrictedImageNet(path=path, split='val',
                                transform=transform, balanced=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                         num_workers=num_workers)
    return loader


def get_restrictedImageNetLabels():
    return RESTRICTED_IMAGNET_LABELS

class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        #np.random.shuffle(indices)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)


#https://github.com/MadryLab/robustness/blob/master/robustness/tools/constants.py
RESTRICTED_IMAGNET_RANGES = [(151, 268), (281, 285),
        (30, 32), (33, 37), (80, 100), (365, 382),
          (389, 397), (118, 121), (300, 319)]

RESTRICTED_IMAGNET_LABELS = [
    'Dog', 'Cat', 'Frog', 'Turtle', 'Bird', 'Monkey', 'Fish', 'Crab', 'Insect'
]
#     * Dog (classes 151-268)
#     * Cat (classes 281-285)
#     * Frog (classes 30-32)
#     * Turtle (classes 33-37)
#     * Bird (classes 80-100)
#     * Monkey (classes 365-382)
#     * Fish (classes 389-397)
#     * Crab (classes 118-121)
#     * Insect (classes 300-319)
class RestrictedImageNet(Dataset):
    def __init__(self, path, split, transform=None, balanced=True):
        self.imagenet = datasets.ImageNet(path, split=split, transform=transform)
        self.balanced = balanced
        self.num_classes = len(RESTRICTED_IMAGNET_RANGES)

        class_idcs = []
        for i in range(self.num_classes):
            class_idcs.append([])

        for i, label in enumerate(self.imagenet.targets):
            for class_idx, (start, end) in enumerate(RESTRICTED_IMAGNET_RANGES):
                if (label >= start) and (label <= end):
                    class_idcs[class_idx].append(i)
                    break

        self.imagenet_linear_idcs = [item for sublist in class_idcs for item in sublist]
        self.targets = [class_idx for class_idx in range(self.num_classes) for _ in class_idcs[class_idx]]

        self.max_class_imgs = max([len(a) for a in class_idcs])
        self.min_class_imgs = min([len(a) for a in class_idcs])

        if not self.balanced:
            self.length = sum([len(a) for a in class_idcs])
        else:
            self.length = self.num_classes * self.min_class_imgs

    def __getitem__(self, index):
        imagenet_linear_idx = self.imagenet_linear_idcs[index]
        target = self.targets[index]
        img, _ = self.imagenet[imagenet_linear_idx]
        return img, target

    def __len__(self):
        return self.length



class RestrictedImagenetBalancedSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled ref_data"""

    def __init__(self, dataset, shuffle):
        assert dataset.balanced
        self.num_classes = dataset.num_classes
        self.class_idcs = []
        targets_tensor = torch.LongTensor(dataset.targets)
        for i in range(self.num_classes):
            class_i_idcs = torch.nonzero(targets_tensor == i, as_tuple=False).squeeze()
            self.class_idcs.append(class_i_idcs)
        self.shuffle = shuffle
        self.min_class_length = min([len(a) for a in self.class_idcs])
        self.length = self.num_classes * self.min_class_length
        super().__init__(None)

    def __iter__(self):
        lin_idcs = torch.zeros(self.length, dtype=torch.long)
        for i in range(self.num_classes):
            s_idx = i * self.min_class_length
            e_idx = (i+1) * self.min_class_length
            if self.shuffle:
                permutation_idcs = torch.randperm(len(self.class_idcs[i]))[:self.min_class_length]
                lin_idcs[s_idx:e_idx] = self.class_idcs[i][permutation_idcs]
            else:
                lin_idcs[s_idx:e_idx] = self.class_idcs[i][:self.min_class_length]

        if self.shuffle:
            lin_idcs = lin_idcs[torch.randperm(len(lin_idcs))]

        return iter(lin_idcs)


    def __len__(self):
        return self.length

def get_restrictedImageNetOD(train=True, batch_size=None, shuffle=None, augm_type='test',
                             num_workers=8, size=224, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'test':
        print('Warning: ImageNet test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_imagenet_path()

    if train == True:
        dataset = RestrictedImageNetOD(path, split='train', transform=transform)
    else:
        dataset = RestrictedImageNetOD(path, split='val', transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'RestrictedImageNetOD'
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader


class RestrictedImageNetOD(torch.utils.data.Dataset):
    def __init__(self, path, split, transform=None):
        self.imagenet = datasets.ImageNet(path, split=split, transform=transform)
        self.num_classes = len(RESTRICTED_IMAGNET_RANGES)

        od_flags = torch.ones(len(self.imagenet), dtype=torch.bool)

        for i, label in enumerate(self.imagenet.targets):
            for class_idx, (start, end) in enumerate(RESTRICTED_IMAGNET_RANGES):
                if (label >= start) and (label <= end):
                    od_flags[i] = False
                    break

        self.od_idcs = torch.nonzero(od_flags, as_tuple=False).squeeze()
        self.length = len(self.od_idcs)

    def __getitem__(self, index):
        imagenet_idx = self.od_idcs[index]
        return self.imagenet[imagenet_idx]

    def __len__(self):
        return self.length


class ImageNetWIDSubset(torch.utils.data.Dataset):
    def __init__(self, path, split, wids, transform=None):
        self.imagenet = datasets.ImageNet(path, split=split, transform=transform)
        self.num_classes = len(wids)

        class_idcs = []
        targets = []
        targets_tensor = torch.LongTensor(self.imagenet.targets)
        for i in range(self.num_classes):
            wid = wids[i]
            wid_idx = self.imagenet.wnid_to_idx[wid]
            wid_bool_idcs = targets_tensor == wid_idx
            wid_lin_idcs = torch.nonzero(wid_bool_idcs, as_tuple=False).squeeze()
            targets.append( i * torch.ones(len(wid_lin_idcs),dtype=torch.long))
            class_idcs.append(wid_lin_idcs)

        self.idcs = torch.cat(class_idcs)
        self.targets = torch.cat(targets)
        self.length = len(self.idcs)
        print(f'ImageNet subset from {self.num_classes} WIDS - Length: {self.length}')


    def __getitem__(self, index):
        imagenet_linear_idx = self.idcs[index]
        target = self.targets[index]
        img, _ = self.imagenet[imagenet_linear_idx]
        return img, target

    def __len__(self):
        return self.length

def get_ImageNet100(train=True, batch_size=None, shuffle=None, augm_type='none',
                    num_workers=8, size=224, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'test' and augm_type != 'none':
        print('Warning: ImageNet test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_imagenet_path()

    if train == True:
        dataset = ImageNetWIDSubset(path, split='train', wids=IMAGENET100_WIDS, transform=transform)
    else:
        dataset = ImageNetWIDSubset(path, split='val', wids=IMAGENET100_WIDS, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'ImageNet100'
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader

def get_ImageNet100_labels():
    return IMAGENET100_LABELS


IMAGENET100_WIDS = ['n02701002', 'n02814533', 'n02930766', 'n03100240', 'n03594945', 'n03670208', 'n03770679',
                    'n03777568', 'n04037443', 'n04285008', 'n02704792', 'n03345487', 'n03417042', 'n03796401',
                    'n03977966', 'n03930630', 'n04461696', 'n04467665', 'n03444034', 'n03445924', 'n03785016',
                    'n04252225', 'n02799071', 'n02802426', 'n03134739', 'n03445777', 'n03942813', 'n04023962',
                    'n04118538', 'n04254680', 'n04409515', 'n04540053', 'n03598930', 'n06785654', 'n03982430',
                    'n07579787', 'n07613480', 'n07614500', 'n07615774', 'n07584110', 'n07590611', 'n07697313',
                    'n07697537', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n09193705', 'n09472597',
                    'n09256479', 'n09421951', 'n09399592', 'n09246464', 'n09288635', 'n09332890', 'n09428293',
                    'n09468604', 'n02676566', 'n03272010', 'n02787622', 'n02992211', 'n04536866', 'n03452741',
                    'n04515003', 'n03495258', 'n03017168', 'n03249569', 'n03447721', 'n03720891', 'n03721384',
                    'n04311174', 'n03042490', 'n03781244', 'n03877845', 'n04613696', 'n03776460', 'n02776631',
                    'n02791270', 'n02871525', 'n02927161', 'n03089624', 'n04200800', 'n04443257', 'n04462240',
                    'n03461385', 'n02687172', 'n04347754', 'n03095699', 'n03673027', 'n03947888', 'n04606251',
                    'n02951358', 'n04612504', 'n03344393', 'n03447447', 'n03662601', 'n04273569', 'n02708093',
                    'n03196217', 'n04548280']

#all wids under node feline and dog
pet_WIDS = ['n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046',
            'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867',
            'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467',
            'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754',
            'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889',
            'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209',
            'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429',
            'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006',
            'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029',
            'n02104365', 'n02107142', 'n02107312', 'n02110627', 'n02105056', 'n02105162', 'n02105251', 'n02105412',
            'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662',
            'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915',
            'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110806', 'n02110958',
            'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706',
            'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02123045', 'n02123159',
            'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925',
            'n02129165', 'n02129604', 'n02130308']

#all wids under node 'n03791235' - motor vehicle, automotive vehicle
#['French loaf', 'meat loaf, meatloaf', 'bagel, beigel', 'pretzel', 'mashed potato', 'bell pepper', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke', 'artichoke, globe artichoke', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple', 'pomegranate', 'dishwasher, dish washer, dishwashing machine', 'refrigerator, icebox', 'washer, automatic washer, washing machine', 'Dutch oven', 'rotisserie', 'espresso maker', 'microwave, microwave oven', 'toaster', 'waffle iron', 'iron, smoothing iron', 'sewing machine', 'vacuum, vacuum cleaner', 'caldron, cauldron', 'coffeepot', 'teapot', 'Crock Pot', 'frying pan, frypan, skillet', 'wok', 'spatula', 'mixing bowl', 'soup bowl', 'Petri dish', 'wooden spoon', 'red wine', 'cup', 'eggnog', 'espresso']
food_101_WIDS = ['n07684084', 'n07871810', 'n07693725', 'n07695742', 'n07711569', 'n07720875', 'n07714571', 'n07714990',
                 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07730033',
                 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592',
                 'n07754684', 'n07760859', 'n07768694', 'n03207941', 'n04070727', 'n04554684', 'n03259280', 'n04111531',
                 'n03297495', 'n03761084', 'n04442312', 'n04542943', 'n03584829', 'n04179913', 'n04517823', 'n02939185',
                 'n03063689', 'n04398044', 'n03133878', 'n03400231', 'n04596742', 'n04270147', 'n03775546', 'n04263257',
                 'n03920288', 'n04597913', 'n07892512', 'n07930864', 'n07932039', 'n07920052']

#all wids under node ['n07555863', 'n03528263', 'n03621049', 'n03154446', 'n07679356', 'n07881800']
#['ambulance', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'cab, hack, taxi, taxicab', 'convertible', 'jeep, landrover', 'limousine, limo', 'minivan', 'Model T', 'racer, race car, racing car', 'sports car, sport car', 'amphibian, amphibious vehicle', 'fire engine, fire truck', 'garbage truck, dustcart', 'moving van', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'pickup, pickup truck', 'tow truck, tow car, wrecker', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'go-kart', 'golfcart, golf cart', 'moped', 'snowplow, snowplough']
cars_WIDS = ['n02701002', 'n02814533', 'n02930766', 'n03100240', 'n03594945', 'n03670208', 'n03770679', 'n03777568', 'n04037443', 'n04285008', 'n02704792', 'n03345487', 'n03417042', 'n03796401', 'n03977966', 'n03930630', 'n04461696', 'n04467665', 'n03444034', 'n03445924', 'n03785016', 'n04252225']


#'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'rapeseed'
flowers_WIDS = ['n11939491', 'n12057211', 'n11879895']

IMAGENET100_LABELS = ['ambulance',
                      'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
                      'cab, hack, taxi, taxicab', 'convertible', 'jeep, landrover', 'limousine, limo', 'minivan',
                      'Model T', 'racer, race car, racing car', 'sports car, sport car',
                      'amphibian, amphibious vehicle', 'fire engine, fire truck', 'garbage truck, dustcart',
                      'moving van', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
                      'pickup, pickup truck', 'tow truck, tow car, wrecker',
                      'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'go-kart',
                      'golfcart, golf cart', 'moped', 'snowplow, snowplough', 'baseball', 'basketball', 'croquet ball',
                      'golf ball', 'ping-pong ball', 'punching bag, punch bag, punching ball, punchball', 'rugby ball',
                      'soccer ball', 'tennis ball', 'volleyball', 'jigsaw puzzle', 'crossword puzzle, crossword',
                      'pool table, billiard table, snooker table', 'plate', 'trifle', 'ice cream, icecream',
                      'ice lolly, lolly, lollipop, popsicle', 'consomme', 'hot pot, hotpot', 'cheeseburger',
                      'hotdog, hot dog, red hot', 'meat loaf, meatloaf', 'pizza, pizza pie', 'potpie', 'burrito', 'alp',
                      'volcano', 'coral reef', 'sandbar, sand bar', 'promontory, headland, head, foreland',
                      'cliff, drop, drop-off', 'geyser', 'lakeside, lakeshore', 'seashore, coast, seacoast, sea-coast',
                      'valley, vale', 'acoustic guitar', 'electric guitar', 'banjo', 'cello, violoncello',
                      'violin, fiddle', 'grand piano, grand', 'upright, upright piano', 'harp', 'chime, bell, gong',
                      'drum, membranophone, tympan', 'gong, tam-tam', 'maraca', 'marimba, xylophone', 'steel drum',
                      'cliff dwelling', 'monastery', 'palace', 'yurt', 'mobile home, manufactured home',
                      'bakery, bakeshop, bakehouse', 'barbershop', 'bookshop, bookstore, bookstall',
                      'butcher shop, meat market', 'confectionery, confectionary, candy store',
                      'shoe shop, shoe-shop, shoe store', 'tobacco shop, tobacconist shop, tobacconist', 'toyshop',
                      'grocery store, grocery, food market, market',
                      'aircraft carrier, carrier, flattop, attack aircraft carrier', 'submarine, pigboat, sub, U-boat',
                      'container ship, containership, container vessel', 'liner, ocean liner', 'pirate, pirate ship',
                      'wreck', 'canoe', 'yawl', 'fireboat', 'gondola', 'lifeboat', 'speedboat', 'analog clock',
                      'digital clock', 'wall clock']

def get_imagenetWID_complement(wids):
    path = get_imagenet_path()
    imagenet = datasets.ImageNet(path, split='val', transform='none')

    imagenet_wids = imagenet.wnids
    complement = []

    for wid in imagenet_wids:
        if wid not in wids:
            complement.append(wid)

    return complement


def get_ImageNetWithout(dataset, train=True, batch_size=None, shuffle=None, augm_type='none',
                    num_workers=8, size=224):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'test' and augm_type != 'none':
        print('Warning: ImageNet test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_imagenet_path()

    if dataset == 'pets':
        dataset_wids = pet_WIDS
    elif dataset == 'food-101':
        dataset_wids = food_101_WIDS
    elif dataset == 'cars':
        dataset_wids = cars_WIDS
    elif dataset == 'flowers':
        dataset_wids = flowers_WIDS
    else:
        raise NotImplementedError()

    od_wids = get_imagenetWID_complement(dataset_wids)

    if train == True:
        dataset = ImageNetWIDSubset(path, split='train', wids=od_wids, transform=transform)
    else:
        dataset = ImageNetWIDSubset(path, split='val', wids=od_wids, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    return loader

def get_ImageNet100OD(train=True, batch_size=None, shuffle=None, augm_type='none',
                    num_workers=8, size=224, config_dict=None):
    if batch_size == None:
        if train:
            batch_size = DEFAULT_TRAIN_BATCHSIZE
        else:
            batch_size = DEFAULT_TEST_BATCHSIZE

    augm_config = {}
    transform = get_imageNet_augmentation(type=augm_type, out_size=size, config_dict=augm_config)
    if not train and augm_type != 'test' and augm_type != 'none':
        print('Warning: ImageNet test set with ref_data augmentation')

    if shuffle is None:
        shuffle = train

    path = get_imagenet_path()

    od_wids = get_imagenetWID_complement(IMAGENET100_WIDS)

    if train == True:
        dataset = ImageNetWIDSubset(path, split='train', wids=od_wids, transform=transform)
    else:
        dataset = ImageNetWIDSubset(path, split='val', wids=od_wids, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

    if config_dict is not None:
        config_dict['Dataset'] = 'ImageNet100OD'
        config_dict['Batch size'] = batch_size
        config_dict['Augmentation'] = augm_config

    return loader
