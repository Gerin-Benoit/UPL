import os
import pickle
from collections import OrderedDict
from tqdm import tqdm

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
from .datasetbase import UPLDatasetBase

from .imagenet import ImageNet


@DATASET_REGISTRY.register()
class SSImageNetV2(UPLDatasetBase):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "images"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)
        self.img_dir = []

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)
        ss_train = self.read_sstrain_data(data)

        super().__init__(train_x=data, test=data, sstrain=ss_train)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
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