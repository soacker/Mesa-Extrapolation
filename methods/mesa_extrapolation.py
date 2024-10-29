"""
  Our implementation uses a plugin-based approach that intercepts calls through monkey patching, enabling seamless integration with existing large models.
"""

import copy

import numpy as np
import torch

from transformers.models.llama.modeling_llama import *


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def chunk_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:


    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict


    if past_key_values == None:

        batch_size, input_length = input_ids.shape

        position_set.set_align_position(input_length)
        first_chunk_length = position_set.first_chunk
        chunk_width = position_set.context_window_length

        new_past_key_values = None
        first_chunk_past_key_values = None
        new_logits = None

        i = 0
        beg, end = 0, 0+first_chunk_length

        while i < input_length:
            outputs = self.model(
                input_ids=input_ids[..., beg:end] if input_ids is not None else None,
                attention_mask=attention_mask[..., beg:end] if attention_mask is not None else None,
                position_ids=position_ids[..., beg:end] if position_ids is not None else None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[..., beg:end, :] if inputs_embeds is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            if self.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()


            current_input_len_q = logits.shape[1]

            if new_past_key_values == None:
                new_past_key_values = outputs.past_key_values
                first_chunk_past_key_values = copy.deepcopy(new_past_key_values)
                new_logits = copy.deepcopy(logits)
            else:
                _past_key_values = []
                for tup, pkv in zip(list(new_past_key_values), list(outputs.past_key_values)):
                    tup_ = (torch.concat([tup[0], pkv[0][:,:,-current_input_len_q:,:]], dim=2), torch.cat([tup[1], pkv[1][:,:,-current_input_len_q:,:]], dim=2))
                    _past_key_values.append(tup_)
                new_past_key_values = tuple(_past_key_values)

                logits = logits.to("cpu")
                new_logits = torch.concat([new_logits, logits], dim=1)
            new_logits = new_logits.to("cpu")

            i = end
            beg = end

            # handle previous chunk
            if end + chunk_width < input_length:
                # concatenate the first chunk
                past_key_values = first_chunk_past_key_values
            # handle the last chunk
            else:
                past_key_values = new_past_key_values

            end += chunk_width

        past_key_values = new_past_key_values
        logits = new_logits
    else:
        # next token inference
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        past_key_values = outputs.past_key_values

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()


    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=None,
        attentions=None,
    )


def attention_chunk_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # save past-key-values without PE info
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 考虑 处理chunk时，和 next-token
        past_key_value = (key_states, value_states) if use_cache else None

        kv_seq_len = key_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # chunk PE params
        train_max_len = position_set.train_length
        push_width = position_set.push_width
        push_pos = (kv_seq_len - 1) % position_set.context_window_length

        if kv_seq_len > train_max_len:
            pos1 = torch.arange(kv_seq_len-push_pos, kv_seq_len, dtype=torch.long).to(position_ids.device)
            last_pos = (kv_seq_len - push_pos) // push_width + 1
            pos2_indices = torch.arange(kv_seq_len - push_pos - last_pos, kv_seq_len - push_pos, dtype=torch.long).to(position_ids.device)
            pos2_repeat = pos2_indices.repeat(push_width)
            sorted_pos2, _ = torch.sort(pos2_repeat)
            rope_position = torch.concat([sorted_pos2, pos1], dim=0).to(position_ids.device)[None,]
        else:
            rope_position = torch.arange(kv_seq_len, dtype=torch.long).to(position_ids.device)[None,]

        position_ids = rope_position.to(position_ids.device)

        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids[:, -query_states.shape[2]:])
        _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin, position_ids[:, -key_states.shape[2]:])

        # scaling log-n
        log_n = (torch.arange(1, key_states.shape[2]+1)[None,][:, None, :, None].log() / np.log(train_max_len)).clip(1).to(query_states.dtype)
        query_states = query_states * log_n[:,:,-query_states.shape[2]:,:].to(query_states.device)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                # fix masking
                attention_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def flash_attention_chunk_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # save past-key-values without PE info
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    kv_seq_len = key_states.shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # chunk PE params
    train_max_len = position_set.train_length
    push_width = position_set.push_width
    push_pos = (kv_seq_len - 1) % position_set.context_window_length

    if kv_seq_len > train_max_len:
        pos1 = torch.arange(kv_seq_len - push_pos, kv_seq_len, dtype=torch.long).to(position_ids.device)
        last_pos = (kv_seq_len - push_pos) // push_width + 1
        pos2_indices = torch.arange(kv_seq_len - push_pos - last_pos, kv_seq_len - push_pos, dtype=torch.long).to(
            position_ids.device)
        pos2_repeat = pos2_indices.repeat(push_width)
        sorted_pos2, _ = torch.sort(pos2_repeat)
        rope_position = torch.concat([sorted_pos2, pos1], dim=0).to(position_ids.device)[None,]
    else:
        rope_position = torch.arange(kv_seq_len, dtype=torch.long).to(position_ids.device)[None,]

    position_ids = rope_position.to(position_ids.device)

    query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin,
                                           position_ids[:, -query_states.shape[2]:])
    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin, position_ids[:, -key_states.shape[2]:])

    log_n = (torch.arange(1, key_states.shape[2] + 1)[None,][:, None, :, None].log() / np.log(train_max_len)).clip(
        1).to(query_states.dtype)
    query_states = query_states * log_n[:, :, -query_states.shape[2]:, :].to(query_states.device)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    with torch.backends.cuda.sdp_kernel(enable_flash=True):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, 0.0,
        )


    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    attn_weights = None

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _chunk_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
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

        # fix masking
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else combined_attention_mask
        )

    return combined_attention_mask


class PositionSet:
    context_window_length: int = 1800

    align_position: int = 1024

    # set last chunk width
    last_context: int = 512

    total_length: int = 0

    train_length: int = 2048

    # first chunk width
    first_chunk: int = 100

    max_token_len: int = 50

    first_chunk_cache: int = 100

    push_width: int = 50

    # implement the DynamicSplit method
    def set_align_position(self, total_length):
        if total_length < self.train_length - self.max_token_len:
            return
        N = (total_length + self.max_token_len - self.last_context - self.first_chunk) // (
                    # fill the space
                    self.train_length - self.first_chunk - self.first_chunk_cache)
        M = (total_length + self.max_token_len - self.last_context - self.first_chunk) % (
                    self.train_length - self.first_chunk - self.first_chunk_cache)

        if M < 200:
            self.context_window_length = self.train_length - self.first_chunk - self.first_chunk_cache
        else:
            chunk_len = (total_length + self.max_token_len - self.last_context - self.first_chunk) // (N + 1)
            N = (total_length + self.max_token_len - self.last_context - self.first_chunk) // chunk_len
            MM = (total_length + self.max_token_len - self.last_context - self.first_chunk) % chunk_len
            assert MM < 200, "non reasonable setting"
            self.context_window_length = chunk_len
        assert self.context_window_length + self.first_chunk < self.train_length
        chunks_length = self.context_window_length * N + self.first_chunk
        last_chunk_length = total_length + self.max_token_len - chunks_length
        print("first chunk:{}, chunk-size:{}, chunks-length:{}, last-chunk-length:{}, N:{}".format(self.first_chunk, self.context_window_length,
                                                                                             chunks_length, last_chunk_length, N))



# 1. set chunk operation
import transformers.models.llama.modeling_llama as modeling_llama_weave
modeling_llama_weave.LlamaForCausalLM.forward = chunk_forward

# 2. set attention operation
# modeling_llama_weave.LlamaAttention.forward = attention_chunk_forward
modeling_llama_weave.LlamaAttention.forward = flash_attention_chunk_forward

# 3. fix mask operation
modeling_llama_weave.LlamaModel._prepare_decoder_attention_mask = _chunk_prepare_decoder_attention_mask

# 4. set Positions
position_set = PositionSet()

