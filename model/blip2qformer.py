"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
from model.blip2 import Blip2Base
from model.help_funcs import pad_and_concat
from model.dist_funs import pl_concat_all_gather



class Blip2Qformer(Blip2Base):
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
    def __init__(
        self,
        ptm,
        lm,
        bert_name,
        plm_name,
        temperature,
        plm_tune='freeze',
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        pool_size=0,
        load_4bit=False,
    ):
        super().__init__()
        self.ptm = ptm
        self.lm = lm
        self.pool_size = pool_size

        
        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder(plm_name, load_4bit)
        self.plm_tune = plm_tune
        if plm_tune == 'freeze':
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm")
        
        self.tokenizer, self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.plm.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.prot_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.ptm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temperature = temperature



    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze() # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            # return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def forward(self, batch):
        prot_batch, text_batch = batch
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        prot_embeds = self.plm(**prot_batch, return_dict=True)
        prot_embeds = prot_embeds.last_hidden_state
        if self.plm_tune == 'freeze':
            prot_embeds = prot_embeds.detach()
        batch_size = prot_embeds.shape[0]
        device = prot_embeds.device

        prot_embeds = self.ln_layer(prot_embeds)
        
        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_batch.attention_mask,
            use_cache=True,
            return_dict=True,
        )
        prot_feats = self.prot_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(**text_batch, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, prot_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(prot_feats, p=2, dim=-1)
        text_feats_all, prot_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(prot_feats) # shape = [B * num_gpus, D]
        sim_p2t, sim_t2p, loss_ptc = self.contrast_global(prot_feats, text_feats, prot_feats_all, text_feats_all, return_sim=True)

        ###============== Image-text Matching ===================###
        loss_ptm = 0
        if self.ptm:
            ## not aggregate global tensor because of their different shapes
            prot_embeds_world = pl_concat_all_gather(prot_embeds)
            prot_mask_world = pl_concat_all_gather(prot_batch.attention_mask)
            text_ids_world = pl_concat_all_gather(text_batch.input_ids)
            text_mask_world = pl_concat_all_gather(text_batch.attention_mask)
            with torch.no_grad():
                rank = dist.get_rank()
                weights_t2p = F.softmax(sim_t2p, dim=1) + 1e-4
                weights_t2p[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(0)
                # weights_t2p.fill_diagonal_(0)
                weights_p2t = F.softmax(sim_p2t, dim=1) + 1e-4
                weights_p2t[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(0)
                # weights_p2t.fill_diagonal_(0)

            # select a negative graph for each text
            prot_embeds_neg = []
            prot_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2p[b], 1).item()
                prot_embeds_neg.append(prot_embeds_world[neg_idx])
                prot_mask_neg.append(prot_mask_world[neg_idx])
            
            prot_embeds_neg = torch.stack(prot_embeds_neg, dim=0)
            prot_mask_neg = torch.stack(prot_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_p2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_mask_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_mask_neg = torch.stack(text_mask_neg, dim=0)

            text_ids_all = torch.cat(
                [text_batch.input_ids, text_batch.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_mask_all = torch.cat(
                [text_batch.attention_mask, text_batch.attention_mask, text_mask_neg], dim=0,
            )

            query_tokens_ptm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_mask_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long, device=device)
            attention_mask_all = torch.cat([query_mask_ptm, text_mask_all], dim=1)

            prot_embeds_all = torch.cat([prot_embeds, prot_embeds_neg, prot_embeds], dim=0)  # pos, neg, pos
            prot_mask_all = torch.cat([prot_batch.attention_mask, prot_mask_neg, prot_batch.attention_mask], dim=0)
            
            output_ptm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_ptm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=prot_embeds_all,
                encoder_attention_mask=prot_mask_all,
                return_dict=True,
            )

            pl_embeddings = output_ptm.last_hidden_state[:, : query_tokens_ptm.size(1), :] # keep query tokens only
            pl_output = self.ptm_head(pl_embeddings)
            logits = pl_output.mean(dim=1)

            ptm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(device)
            loss_ptm = F.cross_entropy(logits, ptm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            ## fix an overflow problem caused by fp16
            enable_autocast = query_output.past_key_values[0][0].dtype == torch.float16
            with torch.cuda.amp.autocast(enable_autocast, dtype=torch.float32):
                decoder_input_ids = text_batch.input_ids.clone()
                decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
                labels = decoder_input_ids.masked_fill(
                    decoder_input_ids == self.tokenizer.pad_token_id, -100
                )
                query_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
                attention_mask = torch.cat([query_mask, text_batch.attention_mask], dim=1)
                lm_output = self.Qformer(
                    decoder_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=query_output.past_key_values,
                    return_dict=True,
                    labels=labels,
                )
                loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_ptc + loss_ptm + loss_lm,
            loss_itc=loss_ptc,
            loss_itm=loss_ptm,
            loss_lm=loss_lm,
        )

    def prot_forward(self, prot_batch):
        prot_embeds = self.plm(**prot_batch, return_dict=True)
        prot_embeds = prot_embeds.last_hidden_state
        if self.plm_tune == 'freeze':
            prot_embeds = prot_embeds.detach()
        prot_embeds = self.ln_layer(prot_embeds)
        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_batch.attention_mask,
            use_cache=True,
            return_dict=True,
        )
        prot_feats = self.prot_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        prot_feats = F.normalize(prot_feats, dim=-1, p=2)
        return prot_feats, prot_embeds

    def text_forward(self, text_batch):
        text_output = self.Qformer.bert(**text_batch, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_ptm(self, prot_embeds, prot_mask, text_ids, text_mask):
        batch_size = prot_embeds.size(0)
        device = prot_embeds.device
        query_tokens = self.query_tokens.expand(batch_size, -1, -1) # shape = [B, Nq, D]
        query_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device) # shape = [B, Nq]
        attention_mask = torch.cat([query_mask, text_mask], dim=1) # shape = [B, Nq + N]
        output_ptm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_mask,
            return_dict=True,
        )
        pl_embeddings = output_ptm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        ptm_logit = self.ptm_head(pl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        ptm_logit = ptm_logit[:, 1] # select the axis of the positive class
        return ptm_logit



