import argparse
import torch
# torch.set_printoptions(profile="full")
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from configs.coopupl_default_config.coopupl_default import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.imagenet_sketch
import datasets.imagenet_v2

import trainers.upltrainer
import trainers.hhzsclip


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.n_shots:
        cfg.DATASET.NUM_TRUE_SHOTS = args.n_shots

    cfg.DATASET.NUM_SHOTS = 16


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.CoOpUPLTrainer = CN()
    cfg.TRAINER.CoOpUPLTrainer.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CoOpUPLTrainer.CSC = False  # class-specific context
    cfg.TRAINER.CoOpUPLTrainer.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CoOpUPLTrainer.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CoOpUPLTrainer.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    if args.lambda_s < 0 or args.lambda_s > 1:
        print("INVALID VALUE FOR LAMBDA_S ({})".format(args.lambda_s))
        exit(-1)
    cfg.TRAINER.CoOpUPLTrainer.LAMBDA_S = args.lambda_s
    cfg.TRAINER.CoOpUPLTrainer.LAMBDA_Q = 1 - args.lambda_s
    # cfg.DATASET.NUM_TRUE_SHOTS = args.n_shots

def setup_cfg(args):

    cfg = get_cfg_default()


    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # 3. hh config
    if args.hh_config_file:
        cfg.merge_from_file(args.hh_config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    if args.dataset == "ssimagenet":
        cfg.OPTIM.MAX_EPOCH = 50
    else:
        epoch_mapping = {0: 50, 1: 50, 2: 100, 4: 100, 8: 200, 16: 200}
        n_epochs = epoch_mapping[args.n_shots]
        cfg.OPTIM.MAX_EPOCH = n_epochs

    cfg.freeze()

    return cfg


def main(args):



    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer_list = []
    for i in range(int(cfg.TRAINER.ENSEMBLE_NUM)):
        trainer = build_trainer(cfg)
        if args.model_dir:
            trainer.load_model_by_id(args.model_dir, epoch=args.load_epoch, model_id=i)
        trainer_list.append(trainer)

        predict_label_dict, predict_conf_dict = trainer.load_from_exist_file(file_path='./analyze_results', 
        model_names=cfg.MODEL.PSEUDO_LABEL_MODELS)
        
        trainer.dm.update_ssdateloader(predict_label_dict, predict_conf_dict)
        trainer.train_loader_sstrain = trainer.dm.train_loader_sstrain
        trainer.transductive_loader = trainer.dm.transductive_loader
        trainer.sstrain_with_id(model_id=i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--dataset',
        type=str,
    )
    parser.add_argument(
        '--hh-config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        '--lambda_s',
        type=float,
        default=0.5,
        help='cross entropy loss weight for support samples'
    )
    parser.add_argument(
        '--lambda_q',
        type=float,
        default=0.5,
        help='cross entropy loss weight for query samples'
    )
    parser.add_argument(
        '--n_shots',
        type=int,
        default=0,
        help='Number of shots for transductive'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )

    args = parser.parse_args()
    main(args)
