import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torchvision

from . import constants

class MedCLIPTextModel(nn.Module):
    def __init__(self,
        bert_type=constants.BERT_TYPE,
        proj_dim = 512,
        proj_bias = False) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        # this tokenizer is actually not used
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # take the average of last four layers
        # last_hidden_states = torch.stack(output['hidden_states'][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        # embed = last_hidden_states.permute(1,0,2,3)
        # embed = embed.mean(1).mean(1) # pooling

        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling

        # let's take only the last hidden layer
        # embed = output['pooler_output']

        embed = self.projection_head(embed)
        return embed

class MedCLIPVisionModel(nn.Module):
    '''
    take resnet50 as backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False) # projection head
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            #print('missing keys:', missing_keys)
            #print('unexpected keys:', unexpected_keys)
            #print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location=torch.device('cpu'))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, **kwargs):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        print("In MedCLIPVisionModel.forward:")
        print("pixel_values shape:", pixel_values.shape if pixel_values is not None else "None")

        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds

class MedCLIPVisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type) #using the txt checkpoints here
        self.projection_head = nn.Linear(768, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location=torch.device('cpu'))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, project=True):
        """
        Args:
            pixel_values: Tensor with shape [batch_size, 3, 224, 224]
        """
        if pixel_values.shape[1] == 1:  # Convert grayscale to RGB
            pixel_values = pixel_values.repeat((1, 3, 1, 1))

        # Forward pass through the model
        output = self.model(pixel_values)

        # Extract the appropriate embedding (likely 'pooler_output')
        if 'pooler_output' in output:
            img_embeds = output['pooler_output']
        else:
            raise ValueError(
                f"The vision model output does not contain the required key 'pooler_output'. "
                f"Available keys: {output.keys()}"
            )

        if project and hasattr(self, 'projection_head'):
            img_embeds = self.projection_head(img_embeds)

        #print(f"Extracted image embeddings shape: {img_embeds.shape}")
        return img_embeds

class MedCLIPModel(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 device=torch.device("cpu")):  # Add device argument with a default value
        super().__init__()
        self.device = device  # Store the device
        text_proj_bias = False
        assert vision_cls in [MedCLIPVisionModel, MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint).to(self.device)  # Move vision model to the device
        self.text_model = MedCLIPTextModel(proj_bias=False).to(self.device)  # Move text model to the device

        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / logit_scale_init_value))).to(self.device)

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location=self.device)
            self.load_state_dict(state_dict)
            print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        import wget
        import zipfile
        pretrained_url = None
        if isinstance(self.vision_model, MedCLIPVisionModel):
            # resnet
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
            if input_dir is None:
                input_dir = './pretrained/medclip-resnet'
        elif isinstance(self.vision_model, MedCLIPVisionModelViT):
            # ViT
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
            if input_dir is None:
                input_dir = './pretrained/medclip-vit'
        else:
            raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

            # download url link
            pretrained_url = requests.get(pretrained_url).text
            filename = wget.download(pretrained_url, input_dir)

            # unzip
            zipf = zipfile.ZipFile(filename)
            zipf.extractall(input_dir)
            zipf.close()
            print('\n Download pretrained model from:', pretrained_url)
        
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME), map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, input_ids=None, attention_mask=None):
        #input_ids = input_ids.cuda()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            #attention_mask = attention_mask.cuda()
            attention_mask = attention_mask.to(self.device)
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, pixel_values=None):
        if pixel_values is None:
            raise ValueError("encode_image received None for pixel_values!")

        # Pass pixel_values to the vision model
        vision_output = self.vision_model(pixel_values=pixel_values)

        # Extract 'img_embeds' or the equivalent key
        if isinstance(vision_output, torch.Tensor):
            img_embeds = vision_output
        elif 'pooler_output' in vision_output:
            img_embeds = vision_output['pooler_output']
        else:
            raise ValueError(
                f"The vision model output does not contain 'pooler_output'. "
                f"Available keys: {vision_output.keys()}"
            )

        # Normalize the embeddings
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                **kwargs):

        if pixel_values is not None:
            img_embeds = self.encode_image(pixel_values=pixel_values)
        else:
            raise ValueError("pixel_values is None before calling encode_image!")

        text_embeds = None  # Initialize text_embeds to None
        if input_ids is not None:
            #print("Passing input_ids to encode_text")
            text_embeds = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
            #print("Encoded text embeddings:", text_embeds.shape)

        logits_per_image = None
        logits_per_text = None
        if text_embeds is not None:
            logits_per_image = self.compute_logits(img_embeds, text_embeds)
            logits_per_text = logits_per_image.t()

        loss = None
        if return_loss and text_embeds is not None:
            loss = self.clip_loss(logits_per_text)

        return {'img_embeds': img_embeds, 'text_embeds': text_embeds,
                'logits': logits_per_image, 'loss_value': loss, 'logits_per_text': logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class PromptClassifier(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs


class SuperviseClassifier(nn.Module):
    '''Take MedCLIP model with linear heads for supervised classification on images.'''

    def __init__(self, vision_model, num_class=14, input_dim=512, mode=None, device=None, **kwargs):
        """
        Args:
            vision_model: The MedCLIP vision model.
            num_class: Number of classes to predict.
            input_dim: The embedding dimension output by the vision model.
            mode: Classification mode ('multiclass', 'multilabel', 'binary').
            device: The device to use (CPU/GPU).
        """
        super().__init__()
        self.model = vision_model
        self.num_class = num_class
        self.mode = mode.lower()
        self.device = device or torch.device("cpu")

        # Define the output layer
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, num_class)  # Match input_dim
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            print("Binary loss is used.")
            self.fc = nn.Linear(input_dim, 1)  # Match input_dim

    def forward(self, pixel_values, labels=None, return_loss=True, **kwargs):
        outputs = defaultdict()

        vision_output = self.model(pixel_values=pixel_values, project=False)

        if isinstance(vision_output, dict):
            #print("Vision model output keys:", vision_output.keys())
            if 'img_embeds' in vision_output:
                img_embeds = vision_output['img_embeds']
            else:
                raise ValueError(
                    f"The vision model output does not contain 'img_embeds'. "
                    f"Available keys: {vision_output.keys()}"
                )
        else:
            img_embeds = vision_output

        logits = self.fc(img_embeds)
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits

        if labels is not None and return_loss:
            labels = labels.float()
            if len(labels.shape) == 1:
                labels = labels.view(-1, 1)
            #labels = labels.float()  # Reshape to [32, 1] to match logits
            #print(f"Logits: {logits.detach().cpu().numpy()}")  # Print logits
            #print(f"Labels: {labels.detach().cpu().numpy()}")  # Print labels
            loss = self.loss_fn(logits, labels)
            #print(f"Loss value: {loss.item()}") 
            outputs['loss_value'] = loss

        return outputs

class PartiallyFixedEmbedding(nn.Module):
    def __init__(self, fixed_weights, num_to_learn):
        super().__init__()
        print(f'{num_to_learn} new tokens added to the embedding layer.')
        self.num_fixed = fixed_weights.size(0)
        self.num_to_learn = num_to_learn
        weight = torch.empty(self.num_fixed+num_to_learn, fixed_weights.size(1))
        weight[:self.num_fixed] = fixed_weights
        self.trainable_weight = nn.Parameter(torch.empty(num_to_learn, fixed_weights.size(1)))
        nn.init.kaiming_uniform_(self.trainable_weight)
        weight[self.num_fixed:] = self.trainable_weight
        self.register_buffer('weight', weight)

    def forward(self, inp):
        self.weight.detach_()
        self.weight[self.num_fixed:] = self.trainable_weight
        return nn.functional.embedding(input=inp,
                                       weight=self.weight,
                                       padding_idx=None,
                                       max_norm=None,
                                       norm_type=2.0,
                                       scale_grad_by_freq=False,
                                       sparse=False)


class PromptTuningClassifier(nn.Module):
    '''take MedCLIP model with prompt tuning
    '''
    def __init__(self, medclip_model, n_context, class_specific_context, num_class, mode, ensemble=True,
                 joint_train_emb=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble
        self.n_context = n_context
        self.class_specific_context = class_specific_context
        self.num_class = num_class
        self.mode = mode
        # calculate number of new context tokens
        if class_specific_context:
            self.n_new_tokens = n_context * num_class
        else:
            self.n_new_tokens = n_context
        # add embeddings for new tokens
        self.prev_n_tokens = self.model.text_model.model.embeddings.word_embeddings.num_embeddings
        self.prev_embeddings = copy.deepcopy(self.model.text_model.model.embeddings.word_embeddings.weight.data)
        if not joint_train_emb:
            self.model.text_model.model.embeddings.word_embeddings = PartiallyFixedEmbedding(
                fixed_weights=self.prev_embeddings,
                num_to_learn=self.n_new_tokens
            )
        else:
            num_old = self.prev_n_tokens
            num_new = self.n_new_tokens
            dim = self.prev_embeddings.shape[1]
            self.model.text_model.model.embeddings.word_embeddings = nn.Embedding(num_old + num_new, dim)
            self.model.text_model.model.embeddings.word_embeddings.weight.data[:num_old] = self.prev_embeddings

        # set loss function
        assert mode.lower() in ['multiclass', 'multilabel', 'binary']
        if mode == 'multilabel':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        return

    def forward(self, pixel_values=None, prompt_inputs=None, labels=None, return_loss=True, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }

        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode in ['multiclass', 'binary']: labels = labels.flatten().long()
            loss = self.loss_fn(class_similarities, labels)
            outputs['loss_value'] = loss

        return outputs
