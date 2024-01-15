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
class SSImageNetA(UPLDatasetBase):

    dataset_dir = "imagenet-adversarial"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.img_dir = []

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        test = self.read_data(classnames)

        sstrain = self.read_sstrain_data(test)
        super().__init__(train_x=test, test=test, sstrain=sstrain)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames):
        split_dir = self.image_dir
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
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
                print(item.impath)
                try:
                    sub_impath = './data/' + item.impath.split('/data/')[1]
                except:
                    sub_impath = item.impath
                if sub_impath in predict_label_dict:
                    new_item = Datum(impath=item.impath, label=predict_label_dict[sub_impath], classname=self._lab2cname[predict_label_dict[sub_impath]])
                    items.append(new_item)
                    self.img_dir.append(item.impath)
        return items