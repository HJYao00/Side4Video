from torch.utils.data._utils.collate import default_collate
from datasets.transforms import *
from datasets.random_erasing import RandomErasing
from RandAugment import RandAugment


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label


class SplitLabel(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return self.worker(img_group), label



def train_augmentation(input_size, flip=True):
    if flip:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            GroupRandomHorizontalFlip(is_flow=False)])
    else:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            # GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip_sth()])


def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    normalize = GroupNormalize(input_mean, input_std)
    if 'something' in config.data.dataset:
        groupscale = GroupScale((256, 320))
    else:
        groupscale = GroupScale(int(scale_size))


    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize])

    if training:
        auto_transform = None
        erase_transform = None
        if config.data.rand_aug:
            auto_transform = create_random_augment(
                input_size=256,
                auto_augment="rand-m7-n4-mstd0.5-inc1",
                interpolation="bicubic"
            )
        if config.data.rand_erase:
            erase_transform = RandomErasing(
                0.25,
                mode='pixel',
                max_count=1,
                num_splits=1,
                device="cpu",
            )           

        train_aug = train_augmentation(
            config.data.input_size,
            flip=False if 'something' in config.data.dataset else True)

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            GroupRandomGrayscale(p=0 if 'something' in config.data.dataset else 0.2),
        ])

        if auto_transform is not None:
            print('=> ########## Using RandAugment!')
            unique = torchvision.transforms.Compose([
                SplitLabel(auto_transform), unique])

        if erase_transform is not None:
            print('=> ########## Using RandErasing!')
            return torchvision.transforms.Compose([
                unique, common, SplitLabel(erase_transform)
            ])
            
        return torchvision.transforms.Compose([unique, common])

    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            GroupCenterCrop(config.data.input_size)])
        return torchvision.transforms.Compose([unique, common])




def randAugment(transform_train, config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train




def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    # print(inputs, flush=True)
    # print(labels, flush=True)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    inputs, labels = (
        default_collate(inputs),
        default_collate(labels),
    )
    return inputs, labels