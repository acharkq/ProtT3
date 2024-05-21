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
from transformers import AutoTokenizer , LlamaForCausalLM
# from model.modeling_llama import LlamaForCausalLM
from opendelta import LoraModel
from opendelta.delta_models.lora import LoraConfig
import torch.nn as nn
try:
    from model.llama_flash_attention import replace_flash_attn_with_original_attn, replace_llama_attn_with_flash_attn
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

class ProteinChatPL(pl.LightningModule):
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
    
    def load_weight(self, path='all_checkpoints/proteinchat_guohan/checkpoint.pth'):
        state_dict = torch.load(path, map_location='cpu')['model']
        print('loading guo han\'s weights')
        self.proj.weight.data.copy_(state_dict['esm_llama_proj.weight'])
        self.proj.bias.data.copy_(state_dict['esm_llama_proj.bias'])

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
        self.llm_name = args.llm_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, use_fast=False, padding_side='right')
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.eos_token_id = self.tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        print(f'loading {args.llm_name}')
        self.llm_model = LlamaForCausalLM.from_pretrained(args.llm_name, torch_dtype=torch.bfloat16)
        self.llm_model.resize_token_embeddings(len(self.tokenizer)) # for the special placeholder token
        if self.llm_tune == 'freeze':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif self.llm_tune == 'full':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True
        elif self.llm_tune == 'lora':
            lora_config = LoraConfig(args.lora_r, args.lora_alpha, args.lora_dropout)
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        elif self.llm_tune == 'mid_lora':
            lora_config = LoraConfig(args.lora_r, args.lora_alpha, args.lora_dropout, modified_modules=["q_proj", "v_proj", 'k_proj', "out_proj", "fc1", "fc2"])
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        else:
            raise NotImplementedError()

        self.proj = nn.Linear(512, self.llm_model.config.hidden_size)
        self.save_hyperparameters(args)
    
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
        with open(os.path.join(self.logger.log_dir, name), 'a', encoding='utf8') as f:
            if q_types is not None:
                for p, t, q in zip(predictions, targets, q_types):
                    line = {'prediction': p, 'target': t, 'q_type': q}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')
            else:
                for p, t in zip(predictions, targets):
                    line = {'prediction': p, 'target': t}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')
    
    def save_add_predictions(self, predictions, targets, log_prefix=''):
        assert len(predictions) == len(targets)
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'
        with open(os.path.join(self.logger.log_dir, name), 'a', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def on_validation_epoch_start(self) -> None:
        if self.enable_flash:
            replace_flash_attn_with_original_attn()
        self.prediction_list0 = []
        self.text_seq_list0 = []
        self.prediction_list1 = []
        self.text_seq_list1 = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if (dataloader_idx % 2) == 0:
            return
            _, _, text_batch = batch
            batch_size = text_batch.input_ids.shape[0]
            loss = self.lm_loss(batch)
            self.log(f"dataloader{dataloader_idx}/val loss", float(loss), batch_size=batch_size, sync_dist=True)
            return loss
        elif (dataloader_idx % 2) == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            prot_batch, prompt_batch, target_dicts = batch
            samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch}
            ###============== Captioning Results ===================###
            predictions = self.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_inference_len,
                min_length=self.min_inference_len,
            )

            ## gather and save the predictions in time
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_target_dicts = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_target_dicts, target_dicts)
            if self.global_rank == 0:
                all_predictions = [i for ii in all_predictions for i in ii]
                all_answers = [i for ii in all_target_dicts for i in ii['answers']]
                all_q_types = [i for ii in all_target_dicts for i in ii['q_types']]
                # self.save_add_predictions(all_predictions, all_targets, 'dataset0')
                self.save_predictions(all_predictions, all_answers, all_q_types, log_prefix=f'dataset{dataloader_idx//2}')
            # if dataloader_idx // 2 == 0:
            #     self.prediction_list0.append(predictions)
            #     self.text_seq_list0.append(target_dicts)
            # elif dataloader_idx // 2 == 1:
            #     self.prediction_list1.append(predictions)
            #     self.text_seq_list1.append(target_dicts)
            # else:
            #     raise NotImplementedError
        else:
            raise NotImplementedError

    # def on_validation_epoch_end(self):
    #     if self.enable_flash:
    #         replace_llama_attn_with_flash_attn()
    #     if (self.current_epoch+1) % self.caption_eval_epoch != 0:
    #         return 
        
    #     predictions0 = [i for ii in self.prediction_list0 for i in ii]
    #     targets0 = [i for ii in self.text_seq_list0 for i in ii]
    #     self.reduce_and_evaluate_captioning(predictions0, targets0, 'dataset0')

    #     if len(self.prediction_list1) > 0:
    #         predictions1 = [i for ii in self.prediction_list1 for i in ii]
    #         targets1 = [i for ii in self.text_seq_list1 for i in ii]
    #         self.reduce_and_evaluate_captioning(predictions1, targets1, 'dataset1')

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
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_inference_len) 
            acc = evaluate_exact_match(all_predictions, all_targets)
            self.log(f"{log_prefix}/acc", acc, sync_dist=False)
            self.log(f"{log_prefix}/bleu2", bleu2, sync_dist=False)
            self.log(f"{log_prefix}/bleu4", bleu4, sync_dist=False)
            self.log(f"{log_prefix}/rouge_1", rouge_1, sync_dist=False)
            self.log(f"{log_prefix}/rouge_2", rouge_2, sync_dist=False)
            self.log(f"{log_prefix}/rouge_l", rouge_l, sync_dist=False)
            self.log(f"{log_prefix}/meteor_score", meteor_score, sync_dist=False)

    def lm_loss(self, batch):
        ## note the prot_batch contains the prompt already
        (prot_embeds, prot_mask), prompt_batch, text_batch = batch
        device = text_batch.input_ids.device
        attention_mask = torch.cat((prot_mask, prompt_batch.attention_mask), dim=1)
        empty_targets = torch.ones(attention_mask.size(), dtype=torch.long).to(device).fill_(-100)
        
        attention_mask = torch.cat((attention_mask, text_batch.attention_mask), dim=1)
        targets = text_batch.input_ids.masked_fill(
            text_batch.input_ids == self.tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        text_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids)
        inputs_embeds = torch.cat((self.proj(prot_embeds), prompt_embeds, text_embeds), dim=1)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        
        batch_size = batch[-1].input_ids.size(0)
        ###============== Overall Loss ===================###
        loss = self.lm_loss(batch)
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
        parser.add_argument('--max_inference_len', type=int, default=36)
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

    def prompt_wrap(self, img_embeds, atts_img, prompt='<s>###Human: <protein><proteinHere></protein> '):
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<proteinHere>')
        p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        p_before_embeds = self.llm_model.get_input_embeddings()(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.llm_model.get_input_embeddings()(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img


    @torch.no_grad()
    def generate(
        self, 
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1
        ):
        prompt_batch = samples['prompt_batch']
        prot_embeds, prot_mask = samples['prot_batch']
        prot_embeds = self.proj(prot_embeds)
        prot_embeds, prot_mask = self.prompt_wrap(prot_embeds, prot_mask)

        text_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        inputs_embeds = torch.cat((prot_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((prot_mask, prompt_batch.attention_mask), dim=1)
        
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        print(output_text)
        return output_text


def evaluate_exact_match(predictions, targets):
    acc = 0
    for prediction, target in zip(predictions, targets):
        if prediction.strip() == target.strip():
            acc += 1
    acc = round(acc / len(predictions) * 100, 2)
    return acc