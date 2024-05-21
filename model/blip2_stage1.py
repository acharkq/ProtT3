import contextlib
import torch
from model.blip2qformer import Blip2Qformer
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from tqdm import tqdm
from model.help_funcs import AttrDict, pad_and_concat
from typing import Any, Dict


class Blip2Stage1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        
        self.args = args
        self.rerank_cand_num = args.rerank_cand_num
        self.blip2qformer = Blip2Qformer(args.ptm, args.lm, args.bert_name, args.plm_name, args.temperature, args.plm_tune, args.num_query_token, args.cross_attention_freq, args.projection_dim, args.pool_size, args.load_4bit)
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
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prot_batch, text_batch = batch
        batch_size = prot_batch.input_ids.shape[0]
        blip2_loss = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log(f"loader{dataloader_idx}/val_loss_ptc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log(f"loader{dataloader_idx}/val_loss_ptm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log(f"loader{dataloader_idx}/val_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)
        self.log(f"loader{dataloader_idx}/val_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)
        return blip2_loss.loss

    
    def get_precision(self, precision):
        if precision in {'16', '16-mixed'}:
            return torch.float16
        elif precision in {'bf16', 'bf16-mixed'}:
            return torch.bfloat16
        elif precision in {'32',}:
            return torch.float32
        else:
            raise NotImplementedError
    
    def retrieval_evaluation_and_log(self, match_dataloader, log_prefix="") -> None:
        with self.maybe_autocast(self.get_precision(self.trainer.precision)):
            ## for onto test set
            p2t_acc, t2p_acc, p2t_rec20, t2p_rec20, \
            p2t_rerank_acc, t2p_rerank_acc, p2t_rerank_rec20, t2p_rerank_rec20, \
            prot_feat_total, text_feat_total, prot_embed_total, prot_mask_total, text_total, text_mask_total = \
                eval_retrieval_inbatch_with_rerank(self.blip2qformer, match_dataloader, self.device)

            self.log(f"{log_prefix}inbatch_p2t_acc", p2t_acc, sync_dist=False)
            self.log(f"{log_prefix}inbatch_t2p_acc", t2p_acc, sync_dist=False)
            self.log(f"{log_prefix}inbatch_p2t_rec20", p2t_rec20, sync_dist=False)
            self.log(f"{log_prefix}inbatch_t2p_rec20", t2p_rec20, sync_dist=False)

            self.log(f"{log_prefix}rerank_inbatch_p2t_acc", p2t_rerank_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_inbatch_t2p_acc", t2p_rerank_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_inbatch_p2t_rec20", p2t_rerank_rec20, sync_dist=False)
            self.log(f"{log_prefix}rerank_inbatch_t2p_rec20", t2p_rerank_rec20, sync_dist=False)
            
            p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, sim_p2t = \
                eval_retrieval_fullset(prot_feat_total, text_feat_total, self.device)
            self.log(f"{log_prefix}fullset_p2t_acc", p2t_acc, sync_dist=False)
            self.log(f"{log_prefix}fullset_t2p_acc", t2p_acc, sync_dist=False)
            self.log(f"{log_prefix}fullset_p2t_rec20", p2t_rec20, sync_dist=False)
            self.log(f"{log_prefix}fullset_t2p_rec20", t2p_rec20, sync_dist=False)

            p2t_acc, p2t_rec20, t2p_acc, t2p_rec20 = \
                eval_retrieval_fullset_for_rerank(self.blip2qformer, sim_p2t, prot_embed_total, prot_mask_total, text_total, text_mask_total, self.rerank_cand_num, self.device)
            self.log(f"{log_prefix}rerank_fullset_p2t_acc", p2t_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_fullset_t2p_acc", t2p_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
            self.log(f"{log_prefix}rerank_fullset_t2p_rec20", t2p_rec20, sync_dist=False)


    def on_validation_epoch_end(self) -> None:
        if self.current_epoch == 0 or (self.current_epoch + 1) % self.args.retrieval_eval_epoch != 0:
            return
        if self.trainer.global_rank == 0:
            ## evaluation for mix dataloaders
            if hasattr(self, 'swiss_test_match_loader') and hasattr(self, 'onto_test_match_loader'):
                self.retrieval_evaluation_and_log(self.swiss_test_match_loader, log_prefix="swiss_test_")
                self.retrieval_evaluation_and_log(self.onto_test_match_loader, log_prefix="onto_test_")
                return
            with self.maybe_autocast(self.get_precision(self.trainer.precision)):
                ## for validation set
                p2t_acc, t2p_acc, p2t_rec20, t2p_rec20, \
                p2t_rerank_acc, t2p_rerank_acc, p2t_rerank_rec20, t2p_rerank_rec20,\
                prot_feat_total, text_feat_total, _, _, _, _ = \
                    eval_retrieval_inbatch_with_rerank(self.blip2qformer, self.val_match_loader, self.device)
                
                self.log("val_inbatch_p2t_acc", p2t_acc, sync_dist=False)
                self.log("val_inbatch_t2p_acc", t2p_acc, sync_dist=False)
                self.log("val_inbatch_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("val_inbatch_t2p_rec20", t2p_rec20, sync_dist=False)

                self.log("rerank_val_inbatch_p2t_acc", p2t_rerank_acc, sync_dist=False)
                self.log("rerank_val_inbatch_t2p_acc", t2p_rerank_acc, sync_dist=False)
                self.log("rerank_val_inbatch_p2t_rec20", p2t_rerank_rec20, sync_dist=False)
                self.log("rerank_val_inbatch_t2p_rec20", t2p_rerank_rec20, sync_dist=False)
                
                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, _ = \
                    eval_retrieval_fullset(prot_feat_total, text_feat_total, self.device)
                self.log("val_fullset_p2t_acc", p2t_acc, sync_dist=False)
                self.log("val_fullset_t2p_acc", t2p_acc, sync_dist=False)
                self.log("val_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("val_fullset_t2p_rec20", t2p_rec20, sync_dist=False)

                ## for test set
                p2t_acc, t2p_acc, p2t_rec20, t2p_rec20, \
                p2t_rerank_acc, t2p_rerank_acc, p2t_rerank_rec20, t2p_rerank_rec20, \
                prot_feat_total, text_feat_total, prot_embed_total, prot_mask_total, text_total, text_mask_total = \
                    eval_retrieval_inbatch_with_rerank(self.blip2qformer, self.test_match_loader, self.device)
                self.log("rerank_test_inbatch_p2t_acc", p2t_rerank_acc, sync_dist=False)
                self.log("rerank_test_inbatch_t2p_acc", t2p_rerank_acc, sync_dist=False)
                self.log("rerank_test_inbatch_p2t_rec20", p2t_rerank_rec20, sync_dist=False)
                self.log("rerank_test_inbatch_t2p_rec20", t2p_rerank_rec20, sync_dist=False)

                self.log("test_inbatch_p2t_acc", p2t_acc, sync_dist=False)
                self.log("test_inbatch_t2p_acc", t2p_acc, sync_dist=False)
                self.log("test_inbatch_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("test_inbatch_t2p_rec20", t2p_rec20, sync_dist=False)
                
                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, sim_p2t = \
                    eval_retrieval_fullset(prot_feat_total, text_feat_total, self.device)
                self.log("test_fullset_p2t_acc", p2t_acc, sync_dist=False)
                self.log("test_fullset_t2p_acc", t2p_acc, sync_dist=False)
                self.log("test_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("test_fullset_t2p_rec20", t2p_rec20, sync_dist=False)

                p2t_acc, p2t_rec20, t2p_acc, t2p_rec20 = \
                    eval_retrieval_fullset_for_rerank(self.blip2qformer, sim_p2t, prot_embed_total, prot_mask_total, text_total, text_mask_total, self.rerank_cand_num, self.device)
                self.log("rerank_test_fullset_p2t_acc", p2t_acc, sync_dist=False)
                self.log("rerank_test_fullset_t2p_acc", t2p_acc, sync_dist=False)
                self.log("rerank_test_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
                self.log("rerank_test_fullset_t2p_rec20", t2p_rec20, sync_dist=False)
                del prot_feat_total, text_feat_total

    def training_step(self, batch, batch_idx):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        prot_batch, text_batch = batch
        batch_size = prot_batch.input_ids.shape[0]
        blip2_loss = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss_ptc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_ptm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)
        self.log("train_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return blip2_loss.loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Stage1")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        parser.add_argument('--ptm', action='store_true', help='use graph-text matching or not', default=True)
        parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=True)

        # evaluation
        parser.add_argument('--rerank_cand_num', type=int, default=128)
        
        # plm
        parser.add_argument('--plm_name', type=str, default='facebook/esm2_t30_150M_UR50D')
        parser.add_argument('--plm_tune', type=str, default='freeze')
        parser.add_argument('--load_4bit', action='store_true', default=False)
        parser.add_argument('--pool_size', type=int, default=0)
        
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
    N = prot_feat.shape[0]
    B = 8
    text_feat = text_feat.to(device)
    sim_p2t = []
    for i in tqdm(range(0, N, B)):
        l_prot_feat = prot_feat[i:i+B].to(device)
        l_sim_q2t = (l_prot_feat.unsqueeze(1) @ text_feat.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [N, D, 1]; output shape = [B, N, num_qs]
        l_sim_p2t, _ = l_sim_q2t.max(-1) # shape = [B, N]
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
    return p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, sim_p2t


@torch.no_grad()
def eval_retrieval_fullset_for_rerank(model, sim_p2t_total, prot_embed_total, prot_mask_total, text_total, text_mask_total, rerank_cand_num, device):
    N = sim_p2t_total.shape[0]
    B = 4    
    rcn = rerank_cand_num ## re-rank candidate numbers
    
    hit_p2t = []
    for i in tqdm(range(0, N, B), desc='re-ranking p2t'):
        sim = sim_p2t_total[i:i+B].to(device)
        rB = sim.shape[0] # real batch size
        topk_sim, topk_idx = sim.topk(k=rcn, dim=1) # shape = [B, rcn]
        topk_idx = topk_idx.cpu()
        prot_embed = prot_embed_total[i:i+B].to(device).repeat_interleave(rcn, 0) # shape = [B * rcn, num_qs, D]
        prot_mask = prot_mask_total[i:i+B].to(device).repeat_interleave(rcn, 0) # shape = [B * rcn, num_qs, D]
        text = text_total[topk_idx].flatten(0,1).to(device) # shape = [B * rcn, text_len]
        text_mask = text_mask_total[topk_idx].flatten(0,1).to(device) # shape = [B * rcn, text_len]
        ptm_sim = model.compute_ptm(prot_embed, prot_mask, text, text_mask).reshape(rB, rcn) ## fixme, using the linear clf's logits directly, without softmax
        sorted_ids = torch.argsort(topk_sim + ptm_sim, descending=True).cpu() # shape = [B, rcn]
        # sorted_ids = torch.argsort(gtm_sim, descending=True).cpu() # shape = [B, rcn]
        sorted_ids = torch.gather(topk_idx, 1, sorted_ids) # mapping to original ids
        hit_p2t.append((sorted_ids == torch.arange(i,i+rB).reshape(-1, 1)).int())
    
    hit_p2t = torch.cat(hit_p2t, dim=0) # shape = [N, rcn]
    # p2t_acc = float((hit_p2t[:, 0]).float().mean())
    # p2t_rec20 = float((hit_p2t[:, :20]).float().sum() / N)
    # print(p2t_acc, p2t_rec20)

    hit_t2p = []
    sim_t2p_total = sim_p2t_total.T
    for i in tqdm(range(0, N, B), desc='re-ranking t2p'):
        sim = sim_t2p_total[i:i+B].to(device)
        rB = sim.shape[0]
        topk_sim, topk_idx = sim.topk(k=rcn, dim=1)
        topk_idx = topk_idx.cpu()
        text = text_total[i:i+B].to(device).repeat_interleave(rcn, 0)
        text_mask = text_mask_total[i:i+B].to(device).repeat_interleave(rcn, 0)
        prot_embed = prot_embed_total[topk_idx].to(device).flatten(0,1)
        prot_mask = prot_mask_total[topk_idx].to(device).flatten(0,1)
        ptm_sim = model.compute_ptm(prot_embed, prot_mask, text, text_mask).reshape(rB, rcn)
        sorted_ids = torch.argsort(topk_sim + ptm_sim, descending=True).cpu() # shape = [B, rcn]
        sorted_ids = torch.gather(topk_idx, 1, sorted_ids)
        hit_t2p.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0]).reshape(-1, 1)).int())
    hit_t2p = torch.cat(hit_t2p, dim=0)
    
    p2t_acc = float((hit_p2t[:, 0]).float().mean())
    p2t_rec20 = float((hit_p2t[:, :20]).float().sum() / N)
    t2p_acc = float((hit_t2p[:, 0]).float().mean())
    t2p_rec20 = float((hit_t2p[:, :20]).float().sum() / N)
    p2t_acc = round(p2t_acc * 100, 2)
    p2t_rec20 = round(p2t_rec20 * 100, 2)
    t2p_acc = round(t2p_acc * 100, 2)
    t2p_rec20 = round(t2p_rec20 * 100, 2)
    return p2t_acc, p2t_rec20, t2p_acc, t2p_rec20


@torch.no_grad()
def eval_retrieval_inbatch_with_rerank(model, dataloader, device=None):
    '''
    include rerank
    '''
    assert isinstance(model, Blip2Qformer)
    pad_token_id = model.tokenizer.pad_token_id
    model.eval()
    p2t_acc = 0
    t2p_acc = 0
    p2t_rec20 = 0
    t2p_rec20 = 0
    allcnt = 0
    
    p2t_rerank_acc = 0
    t2p_rerank_acc = 0
    p2t_rerank_rec20 = 0
    t2p_rerank_rec20 = 0

    prot_feat_total = []  
    text_feat_total = []
    
    prot_embed_total = [] 
    prot_mask_total = []
    
    text_total = []
    text_mask_total = []
    
    for batch in tqdm(dataloader):
        prot_batch, text_batch = batch
        prot_batch, text_batch = prot_batch.to(device), text_batch.to(device)
        text_total.append(text_batch.input_ids)
        text_mask_total.append(text_batch.attention_mask)

        
        prot_feats, prot_embeds = model.prot_forward(prot_batch) # shape = [B, num_qs, D]
        text_feats = model.text_forward(text_batch) # shape = [B, D]
        

        sim_q2t = (prot_feats.unsqueeze(1) @ text_feats.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_p2t, _ = sim_q2t.max(-1) # shape = [B, B]

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
        prot_embed_total.append(prot_embeds.cpu())
        prot_mask_total.append(prot_batch.attention_mask.cpu())

        ## reranking
        prot_embeds = prot_embeds.repeat_interleave(B, 0) # shape = [B * B, prot_len, D]
        prot_mask = prot_batch.attention_mask.repeat_interleave(B, 0) # shape = [B * B, prot_len]
        text_ids = text_batch.input_ids.repeat(B, 1) # shape = [B * B, text_len]
        text_mask = text_batch.attention_mask.repeat(B, 1) # shape = [B * B, text_len]

        ## batched reranking
        batch_size = 64
        ptm_sim = []
        for i in range(0, prot_embeds.shape[0], batch_size):
            ptm_sim_local = model.compute_ptm(prot_embeds[i:i+batch_size], prot_mask[i:i+batch_size], text_ids[i:i+batch_size], text_mask[i:i+batch_size])
            ptm_sim.append(ptm_sim_local)
        ptm_sim = torch.cat(ptm_sim, dim=0).reshape(B, B)

        rerank_sim = sim_p2t + ptm_sim

        ## p2t rerank
        sorted_ids = torch.argsort(rerank_sim, descending=True).cpu() # shape = [B, B]
        hit_p2t = (sorted_ids == torch.arange(B).reshape(-1, 1)).float()
        p2t_rerank_acc += float(hit_p2t[:, 0].sum())
        p2t_rerank_rec20 += float(hit_p2t[:, :20].sum())
        
        ## t2p rerank
        sorted_ids = torch.argsort(rerank_sim.T, descending=True).cpu() # shape = [B, B]
        hit_t2p = (sorted_ids == torch.arange(B).reshape(-1, 1)).float()
        t2p_rerank_acc += float(hit_t2p[:, 0].sum())
        t2p_rerank_rec20 += float(hit_t2p[:, :20].sum())

    prot_feat_total = torch.cat(prot_feat_total, dim=0)
    text_feat_total = torch.cat(text_feat_total, dim=0)
    prot_embed_total = pad_and_concat(prot_embed_total)
    prot_mask_total = pad_and_concat(prot_mask_total)
    text_total = pad_and_concat(text_total, fill_value=pad_token_id)
    text_mask_total = pad_and_concat(text_mask_total)
    # # text_total = torch.cat(text_total, dim=0)
    # text_mask_total = torch.cat(text_mask_total, dim=0)

    p2t_acc = round(p2t_acc/allcnt * 100, 2)
    t2p_acc = round(t2p_acc/allcnt * 100, 2)
    p2t_rec20 = round(p2t_rec20 / allcnt * 100, 2)
    t2p_rec20 = round(t2p_rec20 / allcnt * 100, 2)

    p2t_rerank_acc = round(p2t_rerank_acc / allcnt * 100, 2)
    t2p_rerank_acc = round(t2p_rerank_acc / allcnt * 100, 2)
    p2t_rerank_rec20 = round(p2t_rerank_rec20 / allcnt * 100, 2)
    t2p_rerank_rec20 = round(t2p_rerank_rec20 / allcnt * 100, 2)
    return p2t_acc, t2p_acc, p2t_rec20, t2p_rec20, \
        p2t_rerank_acc, t2p_rerank_acc, p2t_rerank_rec20, t2p_rerank_rec20, \
        prot_feat_total, text_feat_total, prot_embed_total, prot_mask_total, text_total, text_mask_total


