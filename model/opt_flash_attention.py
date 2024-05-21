from typing import List, Optional, Tuple
import logging

import torch
from torch import nn

import transformers
from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from transformers.models.opt.modeling_opt import _make_causal_mask, _expand_mask


def _prepare_decoder_attention_mask_original(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

def forward_original(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
    if attn_weights.dtype == torch.float16:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value


def forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    assert not is_cross_attention, "Cross attention is not supported for flash attention"
    assert past_key_value is None, "past_key_value is not None is not supported for flash attention"
    assert not output_attentions, "output_attentions is not supported for flash attention"

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    ## for flash attention
    flash_shape = (bsz, self.num_heads, tgt_len, self.head_dim)
    query_states = query_states.view(*flash_shape)
    key_states = key_states.view(*flash_shape)
    value_states = value_states.view(*flash_shape)
    qkv = torch.stack([query_states, key_states, value_states], dim=2) # shape = [bsz, num_heads, 3, tgt_len, head_dim]
    qkv = qkv.transpose(1, 3)  # [bsz, tgt_len, 3, num_heads, head_dim]

    key_padding_mask = attention_mask


    assert key_padding_mask is not None
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=self.num_heads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_seqlens, max_s, self.dropout if self.training else 0.0,
        softmax_scale=1, causal=True, return_attn_probs=False
    )

    output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                indices, bsz, tgt_len),
                    'b s (h d) -> b s h d', h=self.num_heads)

    attn_output = self.out_proj(rearrange(output, "b s h d -> b s (h d)"))
    return attn_output, None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_opt_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.opt.modeling_opt.OPTDecoder._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.opt.modeling_opt.OPTAttention.forward = forward

def replace_opt_attn_with_original_attn():
    transformers.models.opt.modeling_opt.OPTDecoder._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_original
    transformers.models.opt.modeling_opt.OPTAttention.forward = forward_original

if __name__ == '__main__':
    ## generate tests to verify the equivalence between forward_original and forward
    import torch.nn as nn
    import math
    class FakeNN(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.scaling = 1 / math.sqrt(2048)
            if False:
                self.q_proj = nn.Linear(2048, 2048)
                self.k_proj = nn.Linear(2048, 2048)
                self.v_proj = nn.Linear(2048, 2048)
                self.out_proj = nn.Linear(2048, 2048)
            else:
                self.q_proj = nn.Identity()
                self.k_proj = nn.Identity()
                self.v_proj = nn.Identity()
                self.out_proj = nn.Identity()

            self.is_decoder = True
            self.num_heads = 2
            self.head_dim = 128
            self.embed_dim = 256
            self.dropout = 0

        def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
            # create causal mask
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = None
            if input_shape[-1] > 1:
                combined_attention_mask = _make_causal_mask(
                    input_shape,
                    inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    past_key_values_length=past_key_values_length,
                )

            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                    inputs_embeds.device
                )
                combined_attention_mask = (
                    expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
                )

            return combined_attention_mask
        
        def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
    fakenn = FakeNN().to(torch.bfloat16).to('cuda:0')

    t_len = 3
    fake_input = torch.randn(2, t_len, fakenn.embed_dim).to(torch.bfloat16).to('cuda:0')
    if False:
        fake_lens = torch.randint(0, t_len, (2,)).to('cuda:0')
        fake_lens = torch.LongTensor([3, 2]).to('cuda:0')
        # fake_lens = torch.ones((2,)).to('cuda:0') * 3
        fake_mask = torch.arange(t_len).unsqueeze(0).to('cuda:0') < fake_lens.unsqueeze(1)
    else:
        fake_mask = torch.randint(0, t_len, (2, t_len)).bool().to('cuda:0')

    fake_mask2 = fakenn._prepare_decoder_attention_mask(fake_mask, (2,t_len), fake_input, 0)
    attn_output0, _, _ = forward_original(fakenn, fake_input, None, None, fake_mask2, None, False)
    attn_output1, _, _ = forward(fakenn, fake_input, None, None, fake_mask, None, False) # shape = [2, 3, 256]
    attn_output0 = attn_output0 * fake_mask.unsqueeze(-1)
    
    print(torch.isclose(attn_output0, attn_output1).all()) 
    print(attn_output0.shape, attn_output1.shape)
    difference = (attn_output0- attn_output1).abs()
    print(difference)
    print(difference.sum())