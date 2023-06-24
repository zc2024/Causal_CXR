import torchvision.transforms as transforms
from dataset.nihdataset import NIHDataset
from utils.cutout import CutoutPIL_
from randaugment import RandAugment
import os.path as osp

import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
    else:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, CutoutPIL_(n_holes=args.n_holes, length=args.length))
        train_data_transform = transforms.Compose(train_data_transform_list)

        test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])


    if args.dataname == 'nih':
        dataset_dir = args.dataset_dir
        nih_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        train_dataset = NIHDataset(
            data_path = dataset_dir,
            input_transform = nih_transform,
            train=True
        )
        val_dataset = NIHDataset(
            data_path=dataset_dir,
            input_transform = nih_transform,
            train=False
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
