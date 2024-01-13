import os
import pickle
from collections import OrderedDict
from tqdm import tqdm

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
from .datasetbase import UPLDatasetBase

from .imagenet import ImageNet

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class ImageNetR(DatasetBase):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train_x=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

@DATASET_REGISTRY.register()
class SSImageNetR(UPLDatasetBase):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.img_dir = []

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)
        sstrain = self.read_sstrain_data(data)

        super().__init__(train_x=data, test=data, sstrain=sstrain)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def read_sstrain_data(self, train, predict_label_dict=None):
        items = []
        if predict_label_dict is None:
            for item in tqdm(train):
                new_item = Datum(impath=item.impath, label=-1, classname=None)
                items.append(new_item)
                self.img_dir.append(item.impath)
        else:
            for item in tqdm(train):
                try:
                    sub_impath = './data/' + item.impath.split('/data/')[1]
                except:
                    sub_impath = item.impath
                if sub_impath in predict_label_dict:
                    new_item = Datum(impath=item.impath, label=predict_label_dict[sub_impath], classname=self._lab2cname[predict_label_dict[sub_impath]])
                    items.append(new_item)
                    self.img_dir.append(item.impath)
        return items