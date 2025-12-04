import logging
import os
import torch
import transformers

from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoConfig
from transformers import set_seed
from typing import Dict, Tuple, cast

from vila_u import conversation as conversation_lib
from vila_u.data import make_supervised_data_module
from vila_u.model import VILAULlamaModel, VILAULlamaConfig
from vila_u.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from vila_u.train.vila_u_trainer import VILAUTrainer
from vila_u.train.args import TrainingArguments, ModelArguments, DataArguments
from vila_u.train.callbacks.autoresume_callback import AutoResumeCallback
from vila_u.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    mprint,
)

# import cotvla dataset and datacollator and action tokenizer
from vila_u.data.dataset import ShardedCoTVLADataset, CoTVLADataCollator
from vila_u.model.language_model.action_tokenizer import ActionTokenizer
import numpy as np


local_rank = None

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "VILA-U"

def make_cotvla_data_module(tokenizer, data_args, training_args, model):
    """
    Creates the CoT-VLA specific dataset and collator.
    """
    # 1. Load Action Stats (binned action file)
    # You might want to add 'action_bins_path' to DataArguments
    action_bins_path = getattr(data_args, "action_bins_path", "./test_data/action_bin_edges.npy") 
    
    print("skipping loading action")
    # print(f"Loading action tokenizer stats from {action_bins_path}...")
    # bin_edges = np.load(action_bins_path)
    # action_stats = {"min": bin_edges[:, 0], "max": bin_edges[:, -1]}
    
    action_tokenizer = ActionTokenizer(
        tokenizer=tokenizer,
        # action_bins=bin_edges
    )

    # 2. Extract the Vision Tower (RQ-VAE) to pass to the dataset
    # VILA-U structure: model -> get_vision_tower() -> vision_tower -> rqvaesiglip
    # Adjust this access path based on exact repo structure if it errors
    vision_tower = model.get_vision_tower()
    data_path = "test_data/rt1_100"
    print(f"Loading CoT Data From: {data_path}")

    # 3. Create Dataset
    train_dataset = ShardedCoTVLADataset(
        data_dir=os.path.join(data_path, "train"), # Path to folder containing .npz shards
        tokenizer=tokenizer,
        data_args=data_args,
        # vision_tower=vision_tower,
        action_tokenizer=action_tokenizer
    )
    eval_dataset = ShardedCoTVLADataset(
        data_dir=os.path.join(data_path, "eval"), # Path to folder containing .npz shards
        tokenizer=tokenizer,
        data_args=data_args,
        # vision_tower=vision_tower,
        action_tokenizer=action_tokenizer
    )

    print(f"Train Dataset length: {len(train_dataset)}")
    print(f"Eval Dataset length: {len(eval_dataset)}")

    print("Printing Train Dataset First Example:")
    print(train_dataset[0])

    # 4. Create Collator
    data_collator = CoTVLADataCollator(
        tokenizer=tokenizer,
        data_args=data_args,
    )
# tokenizer=tokenizer,
#     data_args=data_args,
#     action_tokenizer=action_tokenizer,
# )
    # 5. 
    training_args.sample_lens = [len(train_dataset)]
    training_args.eval_sample_lens = [len(eval_dataset)]

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    # return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

##############
# Code to Freeze part of LM
##############
def freeze_all_llm_layers(model):
    # model.llm is the underlying LLaMA in VILA U
    for p in model.llm.parameters():
        p.requires_grad = False

def unfreeze_last_n_llm_layers(model, last_n=2, unfreeze_lm_head=True):
    # Inspect once in a Python shell if needed: print(model.llm)
    # In LLaMA this is usually model.llm.model.layers
    transformer = model.llm.model
    layers = transformer.layers  # this is a list or ModuleList

    for layer in layers[-last_n:]:
        for p in layer.parameters():
            p.requires_grad = True

    if unfreeze_lm_head and hasattr(model.llm, "lm_head"):
        for p in model.llm.lm_head.parameters():
            p.requires_grad = True

def make_vila_trainable_subset(model, last_n_llm_layers=0):
    # 1. Freeze everything in the LLM
    freeze_all_llm_layers(model)

    # 2. Optionally unfreeze a few top LLM layers
    if last_n_llm_layers > 0:
        unfreeze_last_n_llm_layers(model, last_n=last_n_llm_layers)

    # 3. Keep vision tower frozen
    vt = model.get_vision_tower()
    for p in vt.parameters():
        p.requires_grad = False

    # 4. Train only mm projector (and maybe other small heads)
    mm_proj = model.get_mm_projector()
    for p in mm_proj.parameters():
        p.requires_grad = True

    # You can print stats to confirm
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable / 1e6:.1f}M out of {total / 1e6:.1f}M")

    return model
##############
# Code END to Freeze part of LM
##############

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = cast(Tuple[ModelArguments, DataArguments, TrainingArguments], parser.parse_args_into_dataclasses())
    training_args.run_name = training_args.output_dir.split("/")[-1]
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    set_seed(training_args.seed)

    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
    else:
        resume_from_checkpoint = False
        model_cls = VILAULlamaModel
        config = VILAULlamaConfig.from_pretrained(
            model_args.model_name_or_path,
            resume=resume_from_checkpoint
        )
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path
    
    prepare_config_for_training(config, model_args, training_args, data_args)
    
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    mprint(model)

    model.llm.config.use_cache = False
    model.get_llm().requires_grad_(training_args.tune_language_model)
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")

    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        if isinstance(model.get_vision_tower(), RQVAESIGLIPTransformerVisionTower):
            model.get_vision_tower().vision_tower.rqvaesiglip.eval()
            model.get_vision_tower().vision_tower.rqtransformer.requires_grad_(True)
        else:
            raise NotImplementedError()
        print(f"vision tower {training_args.tune_vision_tower}")
        print(f"mm projector {training_args.tune_mm_projector}")

    if not any([training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]):
        logging.warning(
            "You are not tuning any part of the model. Please check if this is intended."
        )

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = model.tokenizer
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    
    # ==========================================================================
    # ### [NEW] ACTION TOKEN INJECTION START
    # ==========================================================================
    print("Injecting CoT-VLA Action Tokens into Tokenizer...")
    
    # 1. Define the new tokens
    action_tokens = [f"<action_{i}>" for i in range(256)]
    special_action_tokens = ["<action_start>", "<action_end>"]
    
    # 2. Resize and Smart-Init
    # We add them as 'additional_special_tokens' so they are not split by BPE
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"additional_special_tokens": special_action_tokens + action_tokens},
        tokenizer=tokenizer,
        model=model.llm, # VILA-U wraps the core llama in .llm
    )
    
    print(f"Added {len(action_tokens) + 2} action tokens. Vocabulary size: {len(tokenizer)}")
    # ==========================================================================
    # ### [NEW] ACTION TOKEN INJECTION END
    # ==========================================================================

    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.num_video_frames = data_args.num_video_frames
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_use_vi_start_end = data_args.mm_use_vi_start_end = (
            model_args.mm_use_vi_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_vi_start_end = model_args.mm_use_vi_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    print("Before Making COTVLA Data Module")
    # ==========================================================================
    # ### [NEW] REPLACE DATA MODULE CALL
    # ==========================================================================
    # Old:
    # data_module = make_supervised_data_module(
    #     tokenizer=tokenizer,
    #     data_args=data_args,
    #     training_args=training_args,
    # )  
    #   
    data_module = make_cotvla_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model=model
    )
    # ==========================================================================
    print("After Making COTVLA Data Module")

    print("============ Training Arguments =============")
    print(training_args)

    callbacks = [AutoResumeCallback()]
    trainer = VILAUTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
        flush=True,
    )
    print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        flush=True,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"Trainable params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M")

    n = 4
    print(f"Freezing Last {n} Layers of LLM")
    make_vila_trainable_subset(trainer.model, last_n_llm_layers=n)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir
    )

if __name__ == "__main__":
    train()