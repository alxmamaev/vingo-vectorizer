import argparse
import torch
from vingonet.models import MobilenetMAC, MobilenetSPoC
from vingonet.models.MobileNetV2 import MobileNetV2
from vingonet.data import TrainDataloader
from vingonet.train import Trainer
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, RandomSizedCrop
)


def get_augmentations(p=1.0):
    return Compose([
        RandomSizedCrop((250, 600), 224, 224),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=1),
        OneOf([
            MotionBlur(p=.6),
            MedianBlur(blur_limit=3, p=0.6),
            Blur(blur_limit=3, p=0.6),
        ], p=1),
        ShiftScaleRotate(shift_limit=0.0825, scale_limit=0.3, rotate_limit=60, p=1),
        OneOf([
            OpticalDistortion(p=0.5),
            GridDistortion(p=.4),
            IAAPiecewiseAffine(p=0.5),
        ], p=0.8),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.9),
        HueSaturationValue(p=0.3),
    ], p=p)


def get_model(mobilenet_weights_path=None):
    model = MobilenetSPoC()
    if mobilenet_weights_path is not None:
        state_dict = torch.load(mobilenet_weights_path, map_location="cpu")
        mobilenet = MobileNetV2()
        mobilenet.load_state_dict(state_dict)
        model.features = mobilenet.features

    return model

def get_dataloader(datapath, augmentations, batch_size):
    dataloader = TrainDataloader(datapath, augmentations, batch_size=batch_size)

    return dataloader

def get_trainer(model, device, lr):
    trainer = Trainer(model, device=device, lr=lr)

    return trainer

def train(trainer, dataloader, n_epoch, checkpoint_dir, checkpoint_rate,
          validaton_rate, validation_keys, validation_queries):
    trainer.train(dataloader, n_epoch=n_epoch,
                  checkpoint_dir=checkpoint_dir, checkpoint_rate=checkpoint_rate,
                  validation_rate=validaton_rate, val_dataset_path=(validation_keys, validation_queries))

def parse():
    parser = argparse.ArgumentParser(description="Program generate sencepiece tokenizer and embedings from your text")
    parser.add_argument("--datapath", type=str, dest="datapath",
                        help="Path to train dataset", default=None)
    parser.add_argument("--n-epoch", type=int, dest="n_epoch",
                        help="Number of epoch", default=10)
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        help="Batch size", default=10)
    parser.add_argument("--device", type=str, dest="device",
                        help="pytorch acseleration device", default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir",
                        help="path to dir to saving checkpoints", default="./checkpoints")
    parser.add_argument("--checkpoint-rate", type=int, dest="checkpoint_rate",
                        help="how many times checkpoints will be save", default=500)
    parser.add_argument("--log-dir", type=str, dest="log_dir",
                        help="tensorboard logdir", default=None)
    parser.add_argument("--validation-rate", type=int, dest="validation_rate",
                        help="validation rate", default=-1)
    parser.add_argument("--validation-keys", type=str, dest="validation_keys",
                        help="path to validation keys", default=None)
    parser.add_argument("--validation-queries", type=str, dest="validation_queries",
                            help="path to validation queries", default=None)
    parser.add_argument("--mobilenet-weights", type=str, dest="mobilenet_weights",
                            help="pretrained mobilenet weights", default=None)
    parser.add_argument("--lr", type=float, dest="lr",
                            help="learning rate", default=1e-3)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse()

    augmentations = get_augmentations()
    dataloader = get_dataloader(args.datapath, augmentations, args.batch_size)
    model = get_model(args.mobilenet_weights)
    trainer = get_trainer(model, args.device, args.lr)

    train(trainer, dataloader, args.n_epoch, args.checkpoint_dir,
          args.checkpoint_rate, args.validation_rate,
          args.validation_keys, args.validation_queries)
