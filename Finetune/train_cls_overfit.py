import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from methods.cls_model import FinetuneClassifier
import torch
import yaml
import os
import datetime
import argparse
from pytorch_lightning import LightningModule, seed_everything
from datasets.cls_dataset import RSNAImageClsDataset, ChexPertImageClsDataset
from datasets.data_module import DataModule
from datasets.transforms import DataTransforms
from dateutil import tz
import warnings
from collections import OrderedDict

from medclip.modeling_medclip import MedCLIPVisionModelViT, SuperviseClassifier

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
#from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts

import yaml
from torch.utils.data import DataLoader
from medclip.prompts import generate_class_prompts
from transformers import AutoTokenizer

from PIL import Image
#from Finetune.datasets.cls_dataset import RSNAImageClsDataset



#from medclip import MedCLIPModel, MedCLIPVisionModelViT
#from medclip import MedCLIPProcessor
#from medclip import PromptClassifier

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                        message=".*torch.distributed._sharded_tensor.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
                        

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
    parser.add_argument("--dataset", type=str, default="rsna", help="Dataset to use: chexpert, rsna")
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--config', type=str, default='../configs/rsna.yaml', help='Path to config file:chexkpert.yaml, rsna.yaml')
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--data_pct", type=float, default=1, help="Percentage of data to use")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--ckpt_dir', type=str, default='../data/ckpts', help='Directory to save model checkpoints')
    parser.add_argument('--logger_dir', type=str, default='../data/log_output', help='Directory to save logs')
    return parser.parse_args()

def SuperviseImageCollator(mode="binary"):
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        images, labels, paths = zip(*batch)

        images = [transforms.ToTensor()(img) if isinstance(img, Image.Image) else img for img in images]

        return {
            "pixel_values": torch.stack(images),
            "labels": torch.stack(labels),
            "paths": paths 
        }
    return collate_fn



if __name__ == '__main__':
    print()
    print('-----' * 10) 
    seed = 42
    seed_everything(seed)
    args = parse_args()
    config = load_config(args.config)

    if args.dataset == "rsna":
        datamodule = DataModule(
            dataset=RSNAImageClsDataset,
            config=config,
            collate_fn=SuperviseImageCollator(mode="binary"),
            transforms=None,
            data_pct=args.data_pct,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        print("Dataset not supported")
        exit()
        
    device = torch.device("cpu")

    if config['cls']['pretrained']:
        checkpoint_path = config['cls']['checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))
        adjusted_checkpoint = OrderedDict()

        vision_model = MedCLIPVisionModelViT()  
        model = SuperviseClassifier(
            vision_model=vision_model,
            num_class=config['cls']['num_classes'],
            input_dim=512,
            mode="binary", 
            device=device
        ).to(device)

        model_state_dict = model.state_dict()
        #common_keys = set(checkpoint.keys()).intersection(set(model_state_dict.keys()))
        #print(f"Number of common keys between checkpoint and model: {len(common_keys)}")

        for k, v in checkpoint.items(): # MEDCLIP
            new_key = k.replace("vision_model.model", "img_encoder_q.model")  # MEDCLIP
            adjusted_checkpoint[new_key] = v

        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        model.load_state_dict(filtered_checkpoint, strict=False)


    else:
        model = FinetuneClassifier(config)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"FinetuneCLS/{args.dataset}/{extension}")
    logger_dir = os.path.join(args.logger_dir, f"FinetuneCLS/{args.dataset}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    print('ckpt_dir: ', ckpt_dir)
    print('logger_dir:', logger_dir)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    loss_model = torch.nn.BCEWithLogitsLoss()
    loss_model.to(device)
    evaluator = Evaluator(medclip_clf=model, eval_dataloader=val_loader)

    train_objectives = [(train_loader, loss_model, 1.0)]

    from medclip.trainer import Trainer  

    train_config = {
        'batch_size': 1,
        'num_epochs': 1,
        'warmup': 0.1, # the first 10% of training steps are used for warm-up
        'lr': 2e-5,
        'weight_decay': 0.01,
        'eval_batch_size': 1,
        'eval_steps': 500,
        'save_steps': 1000,
    }

    model_save_path = f'./checkpoints/vision_text_pretrain_medclip_SS'
    trainer = Trainer()
    trainer.train(
        model,
        train_objectives=train_objectives,
        warmup_ratio=train_config['warmup'],
        epochs=train_config['num_epochs'],
        optimizer_params={'lr':train_config['lr']},
        output_path=model_save_path,
        evaluation_steps=train_config['eval_steps'],
        weight_decay=train_config['weight_decay'],
        save_steps=train_config['save_steps'],
        evaluator=evaluator
        )
    print('done')