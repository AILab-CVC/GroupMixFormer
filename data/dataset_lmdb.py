from data.folder2lmdb import ImageFolderLMDB

class ImageNet(ImageFolderLMDB):
    def __init__(self, root, transform=None):
        super(ImageNet, self).__init__(root, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
