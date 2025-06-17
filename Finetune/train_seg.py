import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

import segmentation_models_pytorch as smp
from methods.seg_model import FinetuneSegmenter
import torch
import yaml
import os
import datetime
import argparse
from pytorch_lightning import LightningModule, seed_everything
from datasets.seg_dataset import RSNASegmentDataset
from methods.backbones.transformer_seg import SETRModel
from methods.backbones.encoder import ImageEncoder
from datasets.data_module import DataModule
from datasets.transforms import DataTransforms
from dateutil import tz
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                        message=".*torch.distributed._sharded_tensor.*")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
    parser.add_argument("--dataset", type=str, default="rsna", help="Dataset to use: rsna")
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--config', type=str, default='../configs/rsna.yaml', help='Path to config file')
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for dataloader")
    parser.add_argument("--data_pct", type=float, default=1, help="Percentage of data to use")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--ckpt_dir', type=str, default='../data/ckpts', help='Directory to save model checkpoints')
    parser.add_argument('--logger_dir', type=str, default='../data/log_output', help='Directory to save logs')
    return parser.parse_args()

if __name__ == '__main__':
    print()
    print('-----' * 10) 
    seed = 42
    seed_everything(seed)
    args = parse_args()
    config = load_config(args.config)


    if args.dataset == "rsna":
  
        datamodule = DataModule(dataset=RSNASegmentDataset, 
                                config=config, collate_fn=None,
                                transforms=DataTransforms, data_pct=args.data_pct,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print("Dataset not supported, need to develp by yourself")
        exit()


    # import pdb; pdb.set_trace()
    # vit: tate_dict=cls_token, pos_embed, patch_embed.proj.weight, patch_embed.proj.bias,
    #resnet_50: state_dict=model.conv1.weight, model.bn1.weight
    encoder = ImageEncoder(model_name=config["seg"]["backbone"], output_dim=config["seg"]['embed_dim']) 
    
    # Todo: Load the checkpoint for the encoder
    # Initialize the backbone model for segmentation
    if 'vit' in config['seg']['backbone']:
        if config['seg']['pretrained']:
            ckpt = torch.load(config['seg']['checkpoint'])
            img_encoder_q_ckpt = {k.replace("img_encoder_q.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("img_encoder_q.model")}
            missing_keys, unexpected_keys = encoder.load_state_dict(img_encoder_q_ckpt, strict=False)
       
        seg_model = SETRModel(
            patch_size=(16, 16),
            in_channels=3,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64]
        )
        seg_model.encoder_2d.bert_model = encoder
        # freeze the encoder
        for param in seg_model.encoder_2d.bert_model.parameters():
            param.requires_grad = False
    
    elif config['seg']['backbone'] == "resnet_50":
        seg_model = smp.Unet(
            config['seg']['backbone'].replace('_', ''), encoder_weights=None, activation=None) #encoder.conv1.weight, encoder.bn1.weight
        if config['seg']['pretrained']:
            ckpt = torch.load( config['seg']['checkpoint'])
            ckpt_dict = dict()
            # Modify the keys in the loaded checkpoint to align with the U-Net model's encoder
            for k, v in ckpt["state_dict"].items():
                if k.startswith("img_encoder_q.model"):
                    new_k = ".".join(k.split(".")[2:])
                    new_k = new_k.replace("blocks", "layer")
                    ckpt_dict[new_k] = v

            ckpt_dict["fc.bias"] = None
            ckpt_dict["fc.weight"] = None

            import pdb; pdb.set_trace()
            seg_model.encoder.load_state_dict(ckpt_dict)  # seg_model.encoder:conv1.weight, bn1.weight # ckpt_dict: conv1.weight, bn1.weight, bn1.bias,
            # Freeze encoder
            for param in seg_model.encoder.parameters():
                param.requires_grad = False
    
    import pdb; pdb.set_trace()
    model = FinetuneSegmenter(config=config, seg_model=seg_model)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"{'FinetuneSeg'}/{args.dataset}/{extension}")
    logger_dir = os.path.join(args.logger_dir, f"{'FinetuneSeg'}/{args.dataset}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    print('ckpt_dir: ', ckpt_dir)
    print('args.ckpt_dir:', args.ckpt_dir)
    
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        callbacks=callbacks,
        logger=pl.loggers.WandbLogger( project='FinetuneSeg', name=f"{args.dataset}_{args.data_pct}_{extension}"),
        strategy='ddp', #ddp, ddp_spawn
        )

    model.training_steps = model.num_training_steps(trainer, datamodule)

    
    # train
    # import pdb; pdb.set_trace()
    trainer.fit(model, datamodule)
    # test
    trainer.test(model, datamodule, ckpt_path="best")

