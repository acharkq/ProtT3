"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import re
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
from typing import Any, Dict
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from tqdm import tqdm
from lavis.models.blip2_models.blip2 import disabled_train
from model.blip2 import Blip2Base
from model.help_funcs import AttrDict
from model.dist_funs import pl_concat_all_gather


# def pro_trans_tokenizer(text_seqs, **kwargs):
#             text_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in text_seqs]
#             return text_seqs, kwargs

class ProTransTokenizer(BertTokenizer):
    def __call__(self, text_seqs, **kwargs):
        text_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in text_seqs]
        return super().__call__(text_seqs, **kwargs)


class ProtClap(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def init_text_encoder(self, model_name):
        # assert model_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        print(f"bert load {model_name}")
        text_encoder = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return text_encoder, tokenizer
    
    def init_protein_encoder(self, plm_name):
        print(f"plm load {plm_name}")
        plm = BertModel.from_pretrained(plm_name, torch_dtype=torch.bfloat16)
        plm_tokenizer = ProTransTokenizer.from_pretrained(plm_name, do_lower_case=False )

        plm.num_features = plm.config.hidden_size
        ln_layer = nn.LayerNorm(plm.num_features)
        return plm_tokenizer, plm, ln_layer

    def __init__(
        self,
        bert_name,
        plm_name,
        temperature,
        plm_tune=False,
        embed_dim=256,
    ):
        super().__init__()
        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder(plm_name)
        self.plm_tune = plm_tune
        if plm_tune == 'freeze':
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm")
        elif plm_tune == 'full':
            for name, param in self.plm.named_parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError()

        self.text_encoder, self.tokenizer = self.init_text_encoder(bert_name)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(self.plm.config.hidden_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.temperature = temperature

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all):
        '''
        features_graph: shape = [B, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, D]
        '''
        bs = features_graph.size(0)

        sim_g2t = features_graph @ features_text_all.t() # shape = [B, B * num_gpus]
        logits_per_graph = sim_g2t / self.temperature
        sim_t2g = features_text @ features_graph_all.t() # shape = [B, B * num_gpus]
        logits_per_text = sim_t2g / self.temperature
    
        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2
        return loss
    
    def contrast_global_ebm_nce(self, features_graph, features_text, features_graph_all, features_text_all):
        '''
        features_graph: shape = [B, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, D]
        '''
        bs = features_graph.size(0)

        sim_g2t = features_graph @ features_text_all.t() # shape = [B, B * num_gpus]
        logits_per_graph = sim_g2t / self.temperature
        sim_t2g = features_text @ features_graph_all.t() # shape = [B, B * num_gpus]
        logits_per_text = sim_t2g / self.temperature
    
        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        pos_ids = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)
        neg_ids = (pos_ids + bs) % (bs * dist.get_world_size())
        labels = torch.cat([torch.ones(bs, dtype=logits_per_graph.dtype, device=self.device),
                            torch.zeros(bs, dtype=logits_per_graph.dtype, device=self.device)])
        logits_graph = logits_per_graph[torch.arange(bs, dtype=torch.long, device=self.device).repeat(2), torch.cat([pos_ids, neg_ids])]
        logits_text = logits_per_text[torch.arange(bs, dtype=torch.long, device=self.device).repeat(2), torch.cat([pos_ids, neg_ids])]
        
        loss_graph = F.binary_cross_entropy_with_logits(logits_graph, labels)
        loss_text = F.binary_cross_entropy_with_logits(logits_text, labels)
        loss = (loss_graph + loss_text) / 2
        return loss 

    def forward(self, batch):
        prot_batch, text_batch = batch
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        #### prot encoding
        plm_output = self.plm(**prot_batch, return_dict=True)
        prot_feats = plm_output.last_hidden_state[:, 0, :]
        if self.plm_tune == 'freeze':
            prot_feats = prot_feats.detach()
        prot_feats = self.prot_proj(prot_feats)
        prot_feats = F.normalize(prot_feats, p=2, dim=-1)
        prot_feats_all = pl_concat_all_gather(prot_feats) # shape = [B * num_gpus, D]
        
        #### text encoding
        text_output = self.text_encoder(**text_batch, return_dict=True) # shape = [B, n_max, D]
        text_feats = text_output.last_hidden_state[:, 0, :]
        text_feats = self.text_proj(text_feats)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        text_feats_all = pl_concat_all_gather(text_feats)
        
        loss = self.contrast_global(prot_feats, text_feats, prot_feats_all, text_feats_all)
        if True:
            loss2 = self.contrast_global_ebm_nce(prot_feats, text_feats, prot_feats_all, text_feats_all)
            loss = (loss + loss2) / 2
        return loss

    def text_forward(self, text_batch):
        text_output = self.text_encoder(**text_batch, return_dict=True) # shape = [B, n_max, D]
        text_feats = text_output.last_hidden_state[:, 0, :]
        text_feats = self.text_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def prot_forward(self, prot_batch):
        plm_output = self.plm(**prot_batch, return_dict=True)
        prot_feats = plm_output.last_hidden_state[:, 0, :]
        if self.plm_tune == 'freeze':
            prot_feats = prot_feats.detach()
        prot_feats = self.prot_proj(prot_feats)
        prot_feats = F.normalize(prot_feats, p=2, dim=-1)
        return prot_feats 


class PLProtClap(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        
        self.args = args
        self.prot_clap = ProtClap(args.bert_name, args.plm_name, args.temperature, args.plm_tune, args.projection_dim)
        self.save_hyperparameters(args)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        prot_batch, text_batch = batch
        batch_size = prot_batch.input_ids.shape[0]
        loss = self.prot_clap(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss", float(loss), batch_size=batch_size, sync_dist=True)
        return loss
    
    def get_precision(self, precision):
        if precision in {'16', '16-mixed'}:
            return torch.float16
        elif precision in {'bf16', 'bf16-mixed'}:
            return torch.bfloat16
        elif precision in {'32',}:
            return torch.float32
        else:
            raise NotImplementedError
        
    def on_validation_epoch_end(self):
        if self.current_epoch == 0 or (self.current_epoch + 1) % self.args.retrieval_eval_epoch != 0:
            return
        if self.trainer.global_rank == 0:
            with self.maybe_autocast(self.get_precision(self.trainer.precision)):
                ## for validation set
                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, prot_feat_total, text_feat_total = \
                    eval_retrieval_inbatch(self.prot_clap, self.val_match_loader, self.device)
                self.log("val_inbatch_p2t_acc", p2t_acc, sync_dist=False)
                self.log("val_inbatch_t2p_acc", t2p_acc, sync_dist=False)
                self.log("val_inbatch_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("val_inbatch_t2p_rec20", t2p_rec20, sync_dist=False)
                
                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20 = \
                    eval_retrieval_fullset(prot_feat_total, text_feat_total, self.device)
                self.log("val_fullset_p2t_acc", p2t_acc, sync_dist=False)
                self.log("val_fullset_t2p_acc", t2p_acc, sync_dist=False)
                self.log("val_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("val_fullset_t2p_rec20", t2p_rec20, sync_dist=False)

                ## for test set
                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, prot_feat_total, text_feat_total = \
                    eval_retrieval_inbatch(self.prot_clap, self.test_match_loader, self.device)
                self.log("test_inbatch_p2t_acc", p2t_acc, sync_dist=False)
                self.log("test_inbatch_t2p_acc", t2p_acc, sync_dist=False)
                self.log("test_inbatch_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("test_inbatch_t2p_rec20", t2p_rec20, sync_dist=False)

                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20 = \
                    eval_retrieval_fullset(prot_feat_total, text_feat_total, self.device)
                self.log("test_fullset_p2t_acc", p2t_acc, sync_dist=False)
                self.log("test_fullset_t2p_acc", t2p_acc, sync_dist=False)
                self.log("test_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("test_fullset_t2p_rec20", t2p_rec20, sync_dist=False)
                del prot_feat_total, text_feat_total

    def training_step(self, batch, batch_idx):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        prot_batch, text_batch = batch
        batch_size = prot_batch.input_ids.shape[0]
        loss = self.prot_clap(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Stage1")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        
        # plm
        parser.add_argument('--plm_name', type=str, default='facebook/esm2_t30_150M_UR50D')
        parser.add_argument('--plm_tune', type=str, default='full')
        parser.add_argument('--load_4bit', action='store_true', default=False)
        
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--retrieval_eval_epoch', type=int, default=10)
        return parent_parser



@torch.no_grad()
def eval_retrieval_fullset(prot_feat, text_feat, device):    
    '''
    prot_feat: shape = [N, D]
    text_feat: shape = [N, D]
    '''
    N = prot_feat.shape[0]
    B = 32
    text_feat = text_feat.to(device)
    sim_p2t = []
    for i in tqdm(range(0, N, B)):
        l_prot_feat = prot_feat[i:i+B].to(device) # shape = [B, D]
        l_sim_p2t = l_prot_feat @ text_feat.t() # shape = [B, N]
        sim_p2t.append(l_sim_p2t)
    sim_p2t = torch.cat(sim_p2t, dim=0).cpu() # shape = [N, N]
    
    rank_p2t = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_p2t[i:i+B].to(device), descending=True)
        rank_p2t.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_p2t = torch.cat(rank_p2t, dim=0)
    
    rank_t2p = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_p2t.T[i:i+B].to(device), descending=True)
        rank_t2p.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_t2p = torch.cat(rank_t2p, dim=0)
    
    p2t_acc = float((rank_p2t == 0).float().mean())
    p2t_rec20 = float((rank_p2t < 20).float().mean())
    t2p_acc = float((rank_t2p == 0).float().mean())
    t2p_rec20 = float((rank_t2p < 20).float().mean())
    p2t_acc = round(p2t_acc * 100, 2)
    p2t_rec20 = round(p2t_rec20 * 100, 2)
    t2p_acc = round(t2p_acc * 100, 2)
    t2p_rec20 = round(t2p_rec20 * 100, 2)
    return p2t_acc, p2t_rec20, t2p_acc, t2p_rec20



@torch.no_grad()
def eval_retrieval_inbatch(model, dataloader, device=None):
    assert isinstance(model, ProtClap)
    model.eval()

    allcnt = 0
    p2t_acc = 0
    t2p_acc = 0
    p2t_rec20 = 0
    t2p_rec20 = 0
    prot_feat_total = []
    text_feat_total = []

    for batch in tqdm(dataloader):
        prot_batch, text_batch = batch
        prot_batch, text_batch = prot_batch.to(device), text_batch.to(device)
        prot_feats = model.prot_forward(prot_batch) # shape = [B, D]
        text_feats = model.text_forward(text_batch) # shape = [B, D]
        
        sim_p2t = prot_feats @ text_feats.t() # shape = [B, B]

        B = sim_p2t.shape[0]
        sorted_ids = sim_p2t.argsort(descending=True).cpu()
        p2t_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = sim_p2t.T.argsort(descending=True).cpu()
        t2p_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        
        p2t_acc += float((p2t_rank == 0).sum())
        t2p_acc += float((t2p_rank == 0).sum())
        p2t_rec20 += float((p2t_rank < 20).sum())
        t2p_rec20 += float((t2p_rank < 20).sum())

        allcnt += B

        prot_feat_total.append(prot_feats.cpu())
        text_feat_total.append(text_feats.cpu())
    
    prot_feat_total = torch.cat(prot_feat_total, dim=0)
    text_feat_total = torch.cat(text_feat_total, dim=0)
    p2t_acc = round(p2t_acc / allcnt * 100, 2)
    t2p_acc = round(t2p_acc / allcnt * 100, 2)
    p2t_rec20 = round(p2t_rec20 / allcnt * 100, 2)
    t2p_rec20 = round(t2p_rec20 / allcnt * 100, 2)
    return p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, prot_feat_total, text_feat_total


