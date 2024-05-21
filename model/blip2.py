"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn

from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer, BitsAndBytesConfig
from transformers import EsmTokenizer, EsmModel
    

def get_gpu_memory(device=0):
    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total


class Blip2Base(BaseModel):
    # @classmethod
    # def init_tokenizer(cls):
    #     tokenizer = BertTokenizer.from_pretrained('./bert_pretrained/')
    #     tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    #     return tokenizer

    @classmethod
    def init_Qformer(cls, model_name, num_query_token, plm_width, cross_attention_freq=2):
        assert model_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        print("bert load microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        
        encoder_config = BertConfig.from_pretrained(model_name)
        encoder_config.encoder_width = plm_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(model_name, config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer, Qformer, query_tokens
    

    def init_protein_encoder(self, plm_name, load_4bit=False):
        assert plm_name.startswith('facebook/esm2')
        plm_tokenizer = EsmTokenizer.from_pretrained(plm_name)
        if not load_4bit:
            plm = EsmModel.from_pretrained(plm_name, add_pooling_layer=False, torch_dtype=torch.bfloat16)
        else:
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
            outputs = get_gpu_memory(6)
            used_memory = outputs[1]
            if used_memory > 1:
                device_map = {"": 7}
            else:
                device_map = {"": 6}
            plm = EsmModel.from_pretrained(
                plm_name, 
                add_pooling_layer=False,
                quantization_config=quant_config,
                load_in_4bit=True,
                load_in_8bit=False,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )

        plm.num_features = plm.config.hidden_size
        ln_layer = nn.LayerNorm(plm.num_features)
        return plm_tokenizer, plm, ln_layer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

