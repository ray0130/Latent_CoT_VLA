import os
import torch

from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..configuration_vila_u import VILAUConfig
from ..vila_u_arch import VILAUMetaModel, VILAUMetaForCausalLM

from vila_u.constants import (ACTION_START, ACTION_END)

class VILAULlamaConfig(VILAUConfig):
    model_type = "vila_u_llama"


class VILAULlamaModel(VILAUMetaModel, VILAUMetaForCausalLM, PreTrainedModel):
    config_class = VILAULlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: VILAULlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        
        return self.init_vlm(config=config, *args, **kwargs)
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token, 
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )

        return super(VILAULlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token, 
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )  

    def save_checkpoint(self, save_dir, **kwargs):
        """
        Adapter for HF Trainer + Deepspeed.

        When Trainer with deepspeed calls model_wrapped.save_checkpoint(save_dir),
        we just reuse the normal VILA U save_pretrained logic that already saves
        llm, vision_tower, and mm_projector into subfolders.
        """
        print("Saving checkpoint HF function: ")
        print(f"Saving to: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        # If there is a custom save_pretrained on VILAUMetaModel, this will run it.
        # Otherwise fallback to the standard PreTrainedModel.save_pretrained.
        self.save_pretrained(save_dir)
    # import torch

    def build_cot_vla_attention_mask(self, input_ids, pad_mask, dtype=torch.float32):
        """
        input_ids: (B, S) long
        pad_mask: (B, S) bool or 0/1, 1 means not pad
        Returns: additive mask of shape (B, 1, S, S)
        """
        ACTION_START_ID = self.llm.vocab_size - 2 #tokenizer.convert_tokens_to_ids(ACTION_START)
        print("ACTION_START_ID:", ACTION_START_ID)
        print("present?", (input_ids[0] == ACTION_START_ID).any())
        device = input_ids.device
        pad_mask = pad_mask.to(dtype=dtype)   # 1 for real tokens, 0 for pad
        B, S = pad_mask.shape

        # how many tokens at the end should be full attention
        FULL_BLOCK_LEN = 32 * 7 + 1 + 2  # last 32*7 actions + 1 eos

        # base_valid[b, i, j] = 1 only if both i and j are non pad for that batch item
        base_valid = pad_mask.unsqueeze(1) * pad_mask.unsqueeze(2)   # (B, S, S)

        # global causal mask for sequence of length S
        causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.float32))  # (S, S)

        mixed = torch.zeros(B, S, S, device=device, dtype=torch.float32)

        for b in range(B):
            # effective length of this sequence
            seq_len_b = int(pad_mask[b].sum().item())
            if seq_len_b == 0:
                continue  # all pad, nothing to do

            # length of full-attention tail for this sequence
            tail_len = min(FULL_BLOCK_LEN, seq_len_b)
            start_full = seq_len_b - tail_len   # index where full-attention region starts

            # region before the tail: standard causal
            if start_full > 0:
                mixed[b, :start_full, :seq_len_b] = causal[:start_full, :seq_len_b]

            # tail region: full attention within the non pad range
            mixed[b, start_full:seq_len_b, :seq_len_b] = 1.0

        # enforce padding constraint
        mixed = mixed * base_valid  # zero out any attention involving pads
        # print("Mixed mask Logical: ")
        # for bb in range(mixed.shape[0]):
        #     for i in range(mixed.shape[1]):
        #         print(mixed.shape, mixed[bb, i, :])
        # convert to additive 4D mask
        # 1 => allowed => 0
        # 0 => disallowed => large negative
        finfo = torch.finfo(dtype)
        additive = (1.0 - mixed) * finfo.min  # (B, S, S)
        additive = additive.unsqueeze(1)      # (B, 1, S, S)

        return additive

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
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
        # print("Initial Input IDS: ", input_ids.shape, input_ids.min(), input_ids.max())
        # print(input_ids[0])
        # initial_input_ids = input_ids
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
        
        # print("Input embed before repack multimodal data: ")
        # print(inputs_embeds)
        # print("Label before repack multimodal: ", labels.shape, labels[0])
        if self.training:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids

        # print("After repack, New Input embeds: ", new_inputs_embeds.shape, new_inputs_embeds.min(), new_inputs_embeds.max())
        # print(new_inputs_embeds)
        # print("After repack, new Labels ", new_labels.shape)
        # print(new_labels)
        # print("self llm: ", self.llm)
        output_attentions = output_attentions if output_attentions is not None else self.llm.config.output_attentions
        # print("input ids: ", input_ids)
        # print("new attention mask pads:", new_attention_mask)


        # Code to create mixed attention (Causal + Full) mask
        # Currently not in use due to CUDA Error
        # custom_mixed_attention_mask = self.build_cot_vla_attention_mask(initial_input_ids, new_attention_mask, dtype=new_inputs_embeds.dtype)
        
        # print("Output attention shape: ", output_attentions)
        # print("attention mask: ",new_attention_mask.shape, new_attention_mask)
        # print("custom_mixed_attention_mask: ", custom_mixed_attention_mask.shape)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.llm.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.llm.config.use_return_dict

        outputs = self.llm.model(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )

        hidden_states = outputs[0]

        image_hidden_states = []
        image_labels = []
        noimage_labels = []
        # print("hidden state: ", hidden_states)
        for i in range(hidden_states.shape[0]):
            label = new_labels[i]
            # print("LABEL: ", label)
            hidden_state = hidden_states[i]
            label_zero = label[:, 0].clone()
            # print("img label starts: ", self.llm.vocab_size - 4, self.llm.vocab_size - 3, self.llm.vocab_size - 2, self.llm.vocab_size - 1)
            # print("NEW img label starts: ", self.llm.vocab_size - 6, self.llm.vocab_size - 5, self.llm.vocab_size - 4, self.llm.vocab_size - 3)
            # print("label zero:", label_zero)
            im_start_tok_id = self.llm.vocab_size - 6
            im_end_tok_id = self.llm.vocab_size - 5
            video_start_tok_id = self.llm.vocab_size - 4
            video_end_tok_id = self.llm.vocab_size - 3
            # print("NEW img label starts: ", im_start_tok_id, im_end_tok_id, video_start_tok_id, video_end_tok_id)
            if self.config.mm_use_vi_start_end:
                image_start_index = torch.nonzero(torch.eq(label_zero, im_start_tok_id)).squeeze(1)
                image_end_index = torch.nonzero(torch.eq(label_zero, im_end_tok_id)).squeeze(1)
                video_start_index = torch.nonzero(torch.eq(label_zero, video_start_tok_id)).squeeze(1)
                video_end_index = torch.nonzero(torch.eq(label_zero, video_end_tok_id)).squeeze(1)
                image_start_index = torch.cat([image_start_index, video_start_index])
                image_end_index = torch.cat([image_end_index, video_end_index])
            else:
                image_start_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 2)).squeeze(1)
                image_end_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 1)).squeeze(1)

            assert len(image_start_index) == len(image_end_index), f"length of image_start_index is {len(image_start_index)}, length of image_end_index is {len(image_end_index)}"
            # print("FOUND IMAGE STARTINDEX: ", image_start_index, image_end_index)
            if len(image_start_index) > 0:
                for start_idx, end_idx in zip(image_start_index, image_end_index):
                    image_label = label[start_idx+1:end_idx, :]
                    # print("IMAGE LABELS: ", image_label)
                    image_labels.append(image_label)
                    image_hidden_state = hidden_state[start_idx:end_idx-1, :]
                    image_hidden_states.append(image_hidden_state)
                    label_zero[start_idx+1:end_idx] = -100

            noimage_labels.append(label_zero)
        
        # For video
        image_hidden_states_aux = []
        image_labels_aux = []
        # print("Image Hidden State: ", image_hidden_states)
        image_hidden_states_length = [img.shape[0] for img in image_hidden_states]
        # print("Image hidden state length:", image_hidden_states_length)
        image_hidden_states_length_relative = [img // min(image_hidden_states_length) for img in image_hidden_states_length]
        for l in range(len(image_hidden_states_length_relative)):
            if image_hidden_states_length_relative[l] > 1:
                image_hidden_states_aux += torch.split(image_hidden_states[l], min(image_hidden_states_length), dim=0)
                image_labels_aux += torch.split(image_labels[l], min(image_hidden_states_length), dim=0)
            else:
                image_hidden_states_aux.append(image_hidden_states[l])
                image_labels_aux.append(image_labels[l])

        if len(image_hidden_states_aux) > 0:
            image_hidden_states = torch.stack(image_hidden_states_aux, 0)
            image_labels = torch.stack(image_labels_aux, 0)

        noimage_labels = torch.stack(noimage_labels, 0)

        logits = self.llm.lm_head(hidden_states)

        loss_fct = CrossEntropyLoss()

        image_loss = None
        if torch.is_tensor(image_hidden_states):
            if hasattr(self.vision_tower.vision_tower, "rqvaesiglip"):
                # print("image_labels: ", image_labels)
                # print("IMAGE LABEL SHIFTED: ",image_labels - self.llm.vocab_size)
                outs = self.vision_tower.vision_tower.rqtransformer(image_hidden_states, image_labels - self.llm.vocab_size, self.vision_tower.vision_tower.rqvaesiglip)
            else:
                raise NotImplementedError()
            B, seq_len, D, C = outs.shape
            image_logits = outs.reshape(B*seq_len*D, C).contiguous()
            image_labels = image_labels.reshape(B*seq_len*D).contiguous() - self.llm.vocab_size
            image_loss = loss_fct(image_logits, image_labels)

        loss = None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = noimage_labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        if image_loss is not None:
            loss = loss + image_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

AutoConfig.register("vila_u_llama", VILAULlamaConfig)
AutoModel.register(VILAULlamaConfig, VILAULlamaModel)