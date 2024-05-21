"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
# from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from lavis.models.blip2_models.blip2 import disabled_train
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
from opendelta import LoraModel
from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
from transformers import BertTokenizer, BitsAndBytesConfig
from model.help_funcs import hf_enable_gradient_checkpointing
# from peft.tuners.lora import LoraLayer
# from peft import (
#     prepare_model_for_kbit_training,
#     LoraConfig as PeftLoraConfig,
#     get_peft_model,
#     PeftModel
# )

# from opendelta.delta_configs

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

def get_gpu_memory(device=0):
    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input



class Blip2OPT(Blip2Base):
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
        bert_name,
        num_query_token=32,
        cross_attention_freq=2,
        plm_model="facebook/esm2_t30_150M_UR50D",
        plm_tune='freeze',
        llm_name="facebook/galactica-1.3b",
        llm_tune='freeze',
        peft_dir='',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.enbale_gradient_checkpointing = args.enbale_gradient_checkpointing

        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder(plm_model)
        self.plm_tune = plm_tune
        if plm_tune == 'freeze':
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm encoder")
        elif plm_tune == 'lora':
            lora_config = DeltaLoraConfig(args.lora_r, 
                                          args.lora_alpha, 
                                          args.lora_dropout,
                                          modified_modules=["query", "value"])
            self.delta = LoraModel.from_config(lora_config, self.plm)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        else:
            raise NotImplementedError()
        
        self.num_query_token = num_query_token
        _, self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.plm.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        self.llm_model, self.llm_tokenizer = self.load_llm(llm_name, load_4bit=False, enable_gradient_checkpointing=self.enbale_gradient_checkpointing)

        if llm_tune == 'freeze':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True
        elif llm_tune == 'lora':
            lora_config = DeltaLoraConfig(args.lora_r, 
                                          args.lora_alpha, 
                                          args.lora_dropout,)
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        elif llm_tune == 'mid_lora':
            lora_config = DeltaLoraConfig(args.lora_r, args.lora_alpha, args.lora_dropout, modified_modules=["q_proj", "v_proj", 'k_proj', "out_proj", "fc1", "fc2"])
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        elif llm_tune == 'peft_lora':
            config = PeftLoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                # target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, config)
            for name, module in self.llm_model.named_modules():
                if isinstance(module, LoraLayer):
                    if True:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if True and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.llm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)


    def load_llm(self, llm_model, load_4bit=False, enable_gradient_checkpointing=True):
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            ## give a device map that assign all layers to device 0
            outputs = get_gpu_memory(0)
            used_memory = outputs[1]
            if used_memory > 1:
                device_map = {"": 1}
            else:
                device_map = {"": 0}
            llm_model = OPTForCausalLM.from_pretrained(
                llm_model, 
                quantization_config=quant_config,
                load_in_4bit=True,
                load_in_8bit=False,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            llm_model.resize_token_embeddings(len(llm_tokenizer)) ## this will cause bug when 
            llm_model = prepare_model_for_kbit_training(llm_model, use_gradient_checkpointing=True)
        else:
            llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
            llm_model.resize_token_embeddings(len(llm_tokenizer)) ## this will cause bug when 
            if enable_gradient_checkpointing:
                llm_model = hf_enable_gradient_checkpointing(llm_model)
        return llm_model, llm_tokenizer

    def forward(self, batch):
        prot_batch, text_batch = batch
        prot_embeds = self.plm(**prot_batch, return_dict=True)
        prot_embeds = prot_embeds.last_hidden_state
        if self.plm_tune == 'freeze':
            prot_embeds = prot_embeds.detach()
        prot_embeds = self.ln_layer(prot_embeds)
        device = prot_embeds.device
        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_batch.attention_mask,
            return_dict=True,
        )
        prot_tokens = self.opt_proj(query_output.last_hidden_state)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=text_batch.attention_mask.dtype, device=device)
        prot_empty_targets = torch.ones(prot_tokens.shape[:2], dtype=torch.long, device=device).fill_(-100)
        
        targets = text_batch.input_ids.masked_fill(text_batch.input_ids == self.llm_tokenizer.pad_token_id, -100)
        targets = targets.masked_fill(text_batch.token_type_ids == 0, -100)
        targets = torch.cat([prot_empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, inputs_embeds), dim=1)
        attention_mask = torch.cat([prot_mask, text_batch.attention_mask], dim=1)
        
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return loss

    def forwardv3(self, batch):
        prot_batch, prompt_batch, text_batch = batch
        prot_embeds = self.plm(**prot_batch, return_dict=True)
        prot_embeds = prot_embeds.last_hidden_state
        if self.plm_tune == 'freeze':
            prot_embeds = prot_embeds.detach()
        prot_embeds = self.ln_layer(prot_embeds)
        device = prot_embeds.device
        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_batch.attention_mask,
            return_dict=True,
        )
        prot_tokens = self.opt_proj(query_output.last_hidden_state)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=text_batch.attention_mask.dtype, device=device)
        prot_empty_targets = torch.ones(prot_tokens.shape[:2], dtype=torch.long, device=device).fill_(-100)
        empty_targets = torch.ones(prompt_batch.attention_mask.shape, dtype=torch.long, device=device).fill_(-100)
        targets = text_batch.input_ids.masked_fill(text_batch.input_ids == self.llm_tokenizer.pad_token_id, -100)
        targets = torch.cat([prot_empty_targets, empty_targets, targets], dim=1)

        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        inputs_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, prompt_embeds, inputs_embeds), dim=1)
        attention_mask = torch.cat([prot_mask, prompt_batch.attention_mask, text_batch.attention_mask], dim=1)
        
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return loss

    def forwardv2(self, batch):
        prot_batch, prompt_batch, text_batch = batch
        prot_embeds = self.plm(**prot_batch, return_dict=True)
        prot_embeds = prot_embeds.last_hidden_state
        if self.plm_tune == 'freeze':
            prot_embeds = prot_embeds.detach()
        prot_embeds = self.ln_layer(prot_embeds)
        device = prot_embeds.device
        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_batch.attention_mask,
            return_dict=True,
        )
        prot_tokens = self.opt_proj(query_output.last_hidden_state)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=text_batch.attention_mask.dtype, device=device)
        targets = text_batch.input_ids.masked_fill(text_batch.input_ids == self.llm_tokenizer.pad_token_id, -100)

        ### forward prefix
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        prefix_embeds = torch.cat([prot_tokens, prompt_embeds], dim=1)
        prefix_mask = torch.cat([prot_mask, prompt_batch.attention_mask], dim=1)
        prefix_output = self.llm_model.model(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            use_cache=True,
            return_dict=True,
        )

        ## forward decoding
        if False:
            attention_mask = torch.cat([prot_mask, prompt_batch.attention_mask, text_batch.attention_mask], dim=1)
        else:
            attention_mask = text_batch.attention_mask
        print(prefix_output.past_key_values)
        outputs = self.llm_model(
            input_ids=text_batch.input_ids,
            attention_mask=attention_mask,
            past_key_values=prefix_output.past_key_values,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return loss
    
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
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        prot_batch = samples['prot_batch']
        prompt_batch = samples['prompt_batch']
        
        # with self.maybe_autocast():
        prot_embeds = self.plm(**prot_batch, return_dict=True)
        prot_embeds = self.ln_layer(prot_embeds.last_hidden_state)

        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_batch.attention_mask,
            return_dict=True,
        )
        prot_tokens = self.opt_proj(query_output.last_hidden_state)
        
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, prompt_embeds), dim=1)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=prompt_batch.attention_mask.dtype, device=prompt_embeds.device)
        attention_mask = torch.cat([prot_mask, prompt_batch.attention_mask], dim=1)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            use_cache=True,
        )
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text
