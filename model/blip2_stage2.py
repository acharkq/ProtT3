import os
import torch
from model.blip2_opt import Blip2OPT
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
# from peft import LoraConfig, TaskType
from typing import Any, Dict
from model.help_funcs import caption_evaluate, AttrDict
try:
    from model.opt_flash_attention import replace_opt_attn_with_flash_attn, replace_opt_attn_with_original_attn
except ModuleNotFoundError:
    pass


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class Blip2Stage2(pl.LightningModule):
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
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_inference_len = args.max_inference_len
        self.min_inference_len = args.min_inference_len
        self.llm_tune = args.llm_tune
        self.enable_flash = args.enable_flash
        if args.llm_name.find('galactica') >= 0:
            self.blip2 = Blip2OPT(args.bert_name,
                                  args.num_query_token, 
                                  args.cross_attention_freq, 
                                  args.plm_model,
                                  args.plm_tune,
                                  args.llm_name,
                                  args.llm_tune, 
                                  args.peft_dir,  
                                  args)
        else:
            raise NotImplementedError()
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('blip2qformer.')[1]:v for k, v in state_dict.items()}
        self.blip2.load_state_dict(state_dict, strict=False)
        return self
    
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

    def save_predictions(self, predictions, targets, q_types=None, log_prefix=''):
        assert len(predictions) == len(targets)
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'
        with open(os.path.join(self.logger.log_dir, name), 'w', encoding='utf8') as f:
            if q_types is not None:
                for p, t, q in zip(predictions, targets, q_types):
                    line = {'prediction': p, 'target': t, 'q_type': q}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')
            else:
                for p, t in zip(predictions, targets):
                    line = {'prediction': p, 'target': t}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def on_validation_epoch_start(self) -> None:
        if self.enable_flash:
            replace_opt_attn_with_original_attn()
        self.saved_dict_list = []
        self.prediction_list0 = []
        self.target_list0 = []
        self.prediction_list1 = []
        self.target_list1 = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if (dataloader_idx % 2) == 0:
            text_batch = batch[-1]
            batch_size = text_batch.input_ids.shape[0]
            loss = self.blip2(batch)
            ###============== Overall Loss ===================###
            self.log(f"dataloader{dataloader_idx}/val loss", float(loss), batch_size=batch_size, sync_dist=True)
        elif (dataloader_idx % 2) == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            prot_batch, prompt_batch, target_dict = batch
            ###============== Captioning Results ===================###
            samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch}
            predictions = self.blip2.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_inference_len,
                min_length=self.min_inference_len
            )
            target_dict['predictions'] = predictions
            self.saved_dict_list.append(target_dict)

    def gather_dict_results(self, dict_list):
        list_of_dict_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(list_of_dict_list, dict_list)
        dict_list = [i for ii in list_of_dict_list for i in ii] ## dict list, each dict has values that are lists of predictions, etc.
        keys = dict_list[0].keys()
        gathered_dict = {} # each value is a list of predictions, etc.
        for key in keys:
            gathered_dict[key] = [i for d in dict_list for i in d[key]]
        dict_list = []
        for i in range(len(gathered_dict['predictions'])):
            d = {k:gathered_dict[k][i] for k in keys}
            dict_list.append(d)
        return dict_list

    def save_results(self, dict_list, log_prefix=""):
        ## save the results
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'
        with open(os.path.join(self.logger.log_dir, name), 'w', encoding='utf8') as f:
            for d in dict_list:
                f.write(json.dumps(d, ensure_ascii=True) + '\n')

    def on_validation_epoch_end(self):
        if self.enable_flash:
            replace_opt_attn_with_flash_attn()
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        result_list = self.gather_dict_results(self.saved_dict_list)
        ## empty cache
        self.saved_dict_list = []
        
        if self.global_rank == 0:
            self.save_results(result_list, 'dataset0')
            all_predictions = [i['predictions'] for i in result_list]
            all_targets = [i['targets'] for i in result_list]
            
            log_prefix = 'dataset0' ## fixme: this is just a placeholder
            if 'q_types' in result_list[0]:
                ## evaluate protein qa
                pass
            else:
                ## evaluate captioning
                bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                    caption_evaluate(all_predictions, all_targets, self.blip2.llm_tokenizer, self.max_inference_len) 
                acc = evaluate_exact_match(all_predictions, all_targets)
                self.log(f"{log_prefix}/acc", acc, sync_dist=False)
                self.log(f"{log_prefix}/bleu2", bleu2, sync_dist=False)
                self.log(f"{log_prefix}/bleu4", bleu4, sync_dist=False)
                self.log(f"{log_prefix}/rouge_1", rouge_1, sync_dist=False)
                self.log(f"{log_prefix}/rouge_2", rouge_2, sync_dist=False)
                self.log(f"{log_prefix}/rouge_l", rouge_l, sync_dist=False)
                self.log(f"{log_prefix}/meteor_score", meteor_score, sync_dist=False)

    @torch.no_grad()
    def validation_step_old(self, batch, batch_idx, dataloader_idx=0):
        if (dataloader_idx % 2) == 0:
            text_batch = batch[-1]
            batch_size = text_batch.input_ids.shape[0]
            loss = self.blip2(batch)
            ###============== Overall Loss ===================###
            self.log(f"dataloader{dataloader_idx}/val loss", float(loss), batch_size=batch_size, sync_dist=True)
        elif (dataloader_idx % 2) == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            prot_batch, prompt_batch, target_dict = batch
            ###============== Captioning Results ===================###
            samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch}
            predictions = self.blip2.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_inference_len,
                min_length=self.min_inference_len
            )
            if dataloader_idx // 2 == 0:
                self.prediction_list0.append(predictions)
                self.target_list0.append(target_dict)
            elif dataloader_idx // 2 == 1:
                self.prediction_list1.append(predictions)
                self.target_list1.append(target_dict)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def on_validation_epoch_end_old(self):
        if self.enable_flash:
            replace_opt_attn_with_flash_attn()
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        predictions0 = [i for ii in self.prediction_list0 for i in ii]
        targets0 = [i for ii in self.target_list0 for i in ii['answers']]
        if 'q_types' in self.target_list0[0]:
            q_types0 = [i for ii in self.target_list0 for i in ii['q_types']]
            self.reduce_and_evaluate_qa(predictions0, targets0, q_types0, 'dataset0')
        else:
            self.reduce_and_evaluate_captioning(predictions0, targets0, 'dataset0')

        if len(self.prediction_list1) > 0:
            predictions1 = [i for ii in self.prediction_list1 for i in ii]
            targets1 = [i for ii in self.target_list1 for i in ii]
            self.reduce_and_evaluate_captioning(predictions1, targets1, 'dataset1')

    def reduce_and_evaluate_qa(self, predictions, targets, q_types, log_prefix=""):
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_q_types = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        dist.all_gather_object(all_q_types, q_types)
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            all_q_types = [i for ii in all_q_types for i in ii]
            self.save_predictions(all_predictions, all_targets, all_q_types, log_prefix=log_prefix)
            
    def reduce_and_evaluate_captioning(self, predictions, targets, log_prefix=""):
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets, log_prefix)
            ## fixme: I am not sure if the max length is the same as previous experiments
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.blip2.llm_tokenizer, self.max_inference_len) 
            acc = evaluate_exact_match(all_predictions, all_targets)
            self.log(f"{log_prefix}/acc", acc, sync_dist=False)
            self.log(f"{log_prefix}/bleu2", bleu2, sync_dist=False)
            self.log(f"{log_prefix}/bleu4", bleu4, sync_dist=False)
            self.log(f"{log_prefix}/rouge_1", rouge_1, sync_dist=False)
            self.log(f"{log_prefix}/rouge_2", rouge_2, sync_dist=False)
            self.log(f"{log_prefix}/rouge_l", rouge_l, sync_dist=False)
            self.log(f"{log_prefix}/meteor_score", meteor_score, sync_dist=False)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        
        batch_size = batch[-1].input_ids.size(0)
        ###============== Overall Loss ===================###
        loss = self.blip2(batch)
        self.log("loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ProtBlip2")
        # train mode
        parser.add_argument('--save_every_n_epochs', type=int, default=0)

        # Bert
        parser.add_argument('--bert_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--llm_name', type=str, default="facebook/galactica-1.3b")
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_inference_len', type=int, default=128)
        parser.add_argument('--min_inference_len', type=int, default=1)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default='')
        parser.add_argument('--peft_dir', type=str, default='')

        ## plm model
        parser.add_argument('--plm_model', type=str, default='facebook/esm2_t30_150M_UR50D')
        parser.add_argument('--plm_tune', type=str, default='freeze')

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=16)
        parser.add_argument('--lora_dropout', type=int, default=0.1)
        parser.add_argument('--enbale_gradient_checkpointing', action='store_true', default=False)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        return parent_parser



def evaluate_exact_match(predictions, targets):
    acc = 0
    for prediction, target in zip(predictions, targets):
        if prediction.strip() == target.strip():
            acc += 1
    acc = round(acc / len(predictions) * 100, 2)
    return acc