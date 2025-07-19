"""import os
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip import constants

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    config = load_config("configs/pretrain.yaml")
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[config['dataset']['mean']], std=[config['dataset']['std']])
    ])

    # Datasets
    traindata = ImageTextContrastiveDataset(
        datalist=config['dataset']['datalist'],
        imgtransform=transform
    )
    train_collate_fn = ImageTextContrastiveCollator()
    trainloader = DataLoader(
        traindata,
        batch_size=config['train']['batch_size'],
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    # Model
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(device)
    loss_model = ImageTextContrastiveLoss(model).to(device)

    train_objectives = [(trainloader, loss_model, 1)]

    # Save path
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"./checkpoints/pretrain_run_{now}"
    os.makedirs(model_save_path, exist_ok=True)

    # Trainer
    trainer = Trainer()
    trainer.train(
        model=model,
        train_objectives=train_objectives,
        warmup_ratio=config['train']['warmup_ratio'],
        epochs=config['train']['epochs'],
        optimizer_params={'lr': config['train']['lr']},
        output_path=model_save_path,
        evaluation_steps=config['train']['eval_steps'],
        weight_decay=config['train']['weight_decay'],
        save_steps=config['train']['save_steps'],
        eval_dataloader=None,
        use_amp=True
    )

    print("✅ Pretraining done.")

if __name__ == "__main__":
    main()
"""


import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts

import yaml
from torch.utils.data import DataLoader
from Finetune.datasets.cls_dataset import RSNAImageClsDataset


#for config
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config("configs/rsna.yaml") #load rsna config

train_csv = config['dataset']['train_csv']
valid_csv = config['dataset']['valid_csv']
img_size = config['cls']['img_size']
use_medclip_weights = config['cls']['pretrained']
checkpoint_path = config['cls']['checkpoint']
freeze_backbone = config['cls']['freeze']


# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# set training configurations
train_config = {
    'batch_size': 64,
    'num_epochs': 1,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 2e-5,
    'weight_decay': 0.01,
    'eval_batch_size': 64,
    'eval_steps': 500,
    'save_steps': 1000,
}

# only pretrain on chexpert train data and mimic-cxr data
# do zero-shot training on chexpert-5x200 and iuxray
datalist = [
    'chexpert-train',
    'mimic-cxr-train',
]

#data augmentation
transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2,0.2),
                transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((config['cls']['img_size'], config['cls']['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304]),
])

traindata = ImageTextContrastiveDataset(datalist=datalist, imgtransform=transform)

train_collate_fn = ImageTextContrastiveCollator()

trainloader = DataLoader(traindata,
    batch_size=train_config['batch_size'],
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=12,
    )

#new
train_dataset = RSNAImageClsDataset(config=config, split='train', transform=transform)
val_dataset = RSNAImageClsDataset(config=config, split='valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModelViT

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
if config['cls']['pretrained']:
    state_dict = torch.load(config['cls']['checkpoint'], map_location='cpu')
    model.load_state_dict(state_dict)

if config['cls']['freeze']:
    for param in model.vision_model.parameters():
        param.requires_grad = False

# Klassifikationskopf
clf_model = PromptClassifier(
    model,
    mode='binary' if config['cls']['num_classes'] == 2 else 'multiclass',
    embed_dim=config['cls']['embed_dim'],
    dropout=config['cls']['dropout']
).cuda()
#end new

# build medclip model
#model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

if use_medclip_weights and checkpoint_path:
    print(f"Loading pretrained weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

if freeze_backbone:
    for param in model.vision_model.parameters():
        param.requires_grad = False

# Wrap with classifier
clf_model = PromptClassifier(model, mode='binary', embed_dim=config['cls']['embed_dim'], dropout=config['cls']['dropout'])
clf_model.cuda()

model.cuda()

# build evaluator
cls_prompts = generate_chexpert_class_prompts(n=10)
val_data = ZeroShotImageDataset(['chexpert-5x200-val'],
    class_names=constants.CHEXPERT_COMPETITION_TASKS)
val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
    mode='multiclass')
eval_dataloader = DataLoader(val_data,
    batch_size=train_config['eval_batch_size'],
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)
medclip_clf = PromptClassifier(model)
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
    mode='multiclass',
)

# build loss models and start training
loss_model = ImageTextContrastiveLoss(model)
loss_model.cuda()
train_objectives = [
    (trainloader, loss_model, 1),
]
model_save_path = f'./checkpoints/vision_text_pretrain'
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
    evaluator=evaluator,
    eval_dataloader=eval_dataloader,
    use_amp=True,
    )
print('done')
