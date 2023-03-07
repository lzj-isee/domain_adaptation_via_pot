import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class myImageFolder(data.Dataset):
    def __init__(self, images, labels, transform = None, target_transform = None, if_return_idx = False):
        self.imgs_path, self.labels = images, labels
        self.transform, self.target_transform = transform, target_transform
        self.if_return_idx = if_return_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.imgs_path[index]
        target = self.labels[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.if_return_idx:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.labels)
