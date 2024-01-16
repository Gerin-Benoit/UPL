from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset
from dassl.utils import read_image
from dassl.data.transforms import build_transform
from dassl.data.transforms.transforms import INTERPOLATION_MODES
from dassl.data.samplers import build_sampler

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

def build_transductive_loader(
    cfg,
    sampler_type="RandomSampler",
    sampler=None,
    data_s=None,
    data_q=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm_s=None,
    tfm_q=None,
    is_train=True,
    dataset_wrapper=None,
    tag=None
):
    # Build sample

    if sampler_type is not None:
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_s+data_q,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=n_ins,
        )
    else:
        sampler = sampler

    if dataset_wrapper is None:
        dataset_wrapper = TransductiveDatasetWrapper

    # Build data loader
    if tag is None:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_s=data_s, data_q=data_q, transform_s=tfm_s, transform_q=tfm_q, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_s + data_q) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_s=data_s, data_q=data_q, transform_s=tfm_s, transform_q=tfm_q, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_s + data_q) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )




    return data_loader

def build_data_loader(
    cfg,
    sampler_type="RandomSampler",
    sampler=None,
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    tag=None
):
    # Build sample

    if isinstance(data_source, tuple):
        merged = []
        for ds in data_source:
            merged += ds
        data_source = merged

    if sampler_type is not None:
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_source,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=n_ins,
        )
    else:
        sampler = sampler

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    if tag is None:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )




    return data_loader

class UPLDataManager(DataManager):
    def __init__(self,
                cfg,
                custom_tfm_train=None,
                custom_tfm_test=None,
                dataset_wrapper=None):
        super().__init__(cfg, custom_tfm_train, custom_tfm_test, dataset_wrapper)

        
        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test
        
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        # save cfg 
        self.cfg = cfg
        self.tfm_train = tfm_train
        self.dataset_wrapper = dataset_wrapper
        self.transductive_loader = None


        if cfg.DATALOADER.OPEN_SETTING:
            test_novel_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.dataset.novel,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

            test_base_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.dataset.base,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

            self.test_novel_loader = test_novel_loader
            self.test_base_loader = test_base_loader
        
        try:
            if self.dataset.sstrain:
                # 除了dataset的source是不一样的，其他跟trian都是一样的
                train_loader_sstrain = build_data_loader(
                    cfg,
                    sampler_type="SequentialSampler",
                    data_source=self.dataset.sstrain,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=dataset_wrapper,
                    tag='sstrain' # 初始化的时候需要设置这个来保证所有样本的载入
                )
                self.train_loader_sstrain = train_loader_sstrain

                # Build train_loader_x
                train_loader_x = build_data_loader(
                    cfg,
                    sampler_type="SequentialSampler",
                    data_source=self.dataset.train_x,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    tfm=tfm_test, # 这个是不训练的，所以要用测试的配置，
                    is_train=False,
                    dataset_wrapper=dataset_wrapper,
                    tag='x'
                )
                self.train_loader_x = train_loader_x
        except:
            pass
        
    def update_ssdateloader(self, predict_label_dict, predict_conf_dict):
        """update the train_loader_sstrain to add labels

        Args:
            predict_label_dict ([dict]): [a dict {'imagepath': 'label'}]
        """
    

        sstrain = self.dataset.add_label(predict_label_dict, self.cfg.DATASET.NAME)
        print('sstrain', len(sstrain))
        
        
        # train_sampler = WeightedRandomSampler(weights, len(sstrain))
        train_loader_sstrain = build_data_loader(
            self.cfg,
            sampler_type="RandomSampler", 
            sampler=None,
            data_source=sstrain,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=1, # 每个类别n_ins个instance
            tfm=self.tfm_train,
            is_train=False,
            dataset_wrapper=self.dataset_wrapper,
        )
        self.train_loader_sstrain = train_loader_sstrain

        # modify datum to add label_type

        transductive_loader = build_transductive_loader(
            self.cfg,
            sampler_type="RandomSampler",
            sampler=None,
            data_s=self.dataset.train_x,
            data_q=sstrain,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=1,
            tfm_s=self.tfm_train,
            tfm_q=self.tfm_train,
            is_train=False,
            dataset_wrapper=None,
        )
        print("=== Size of S, Size of Q : ===")
        print(len(self.dataset.train_x), len(sstrain))
        self.transductive_loader = transductive_loader


class TransductiveDatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_s, data_q, transform_s=None, transform_q=None, is_train=False):
        self.cfg = cfg
        self.data_s = data_s
        self.data_q = data_q
        self.s_size = len(data_s)
        self.q_size = len(data_q)
        self.label_s = "s"
        self.label_q = "q"
        self.transform_s = transform_s  # accept list (tuple) as input
        self.transform_q = transform_q  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and (transform_s is None or transform_q is None):
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform for S or Q is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return self.s_size + self.q_size

    def __getitem__(self, idx):

        if idx < self.s_size:
            item = self.data_s[idx]
            label_type = self.label_s
            transform = self.transform_s
        else:
            item = self.data_q[idx-self.s_size]
            label_type = self.label_q
            transform = self.transform_q

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx,
            "label_type": label_type
        }

        img0 = read_image(item.impath)

        if transform is not None:
            if isinstance(transform, (list, tuple)):
                for i, tfm in enumerate(transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img