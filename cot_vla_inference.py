import argparse
import cv2
import numpy as np
import os
import vila_u


from vila_u.data.dataset import ShardedCoTVLADataset, CoTVLADataCollator
from vila_u.model.language_model.action_tokenizer_sepdim import ActionTokenizer
from transformers import HfArgumentParser, AutoConfig

from vila_u.train.args import TrainingArguments, ModelArguments, DataArguments
from typing import Dict, Tuple, cast, Optional

from transformers import CLIPImageProcessor

from torch.utils.data import DataLoader
from PIL import Image, ImageFile

import time
import torch
from dataclasses import dataclass

from vila_u.constants import ACTION_START, ACTION_END


def save_image(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)


def save_video(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        video = response[i].permute(0, 2, 3, 1)
        video = video.cpu().numpy().astype(np.uint8)
        video = np.concatenate(video, axis=1)
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"video_{i}.png"), video)


@torch.no_grad()
def eval_teacher_forcing(
    model,
    dataloader,
    tokenizer,
    mode,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Teacher-forcing evaluation:
      - Uses GT input_ids/labels from dataset (no generate()).
      - Reads out.logits and evaluates token-level accuracy over:
          (1) action tokens between <action_start> and <action_end>
          (2) boundary tokens: <action_start>, <action_end>
      - Also reports avg loss (whatever model.forward returns).
    """

    model.eval()
    model_dtype = next(model.parameters()).dtype
    total_img_loss = 0.0
    total_act_loss = 0.0
    n_batches = 0
    accuracy = []
    accuracy_start = []
    accuracy_end = []
    mse = []
    mse_per_dim = []
    valid = 0

    print(f"ACTION_START={ACTION_START}, ACTION_END={ACTION_END}")
    ACTION_START_ID = tokenizer.convert_tokens_to_ids(ACTION_START)
    ACTION_END_ID   = tokenizer.convert_tokens_to_ids(ACTION_END)
    print(f"ACTION_START_ID={ACTION_START_ID}, ACTION_END_ID={ACTION_END_ID}")

    # -------- stats --------
    correct_boundary = 0
    total_boundary = 0

    correct_action = 0
    total_action = 0

    correct_all = 0
    total_all = 0  # over all non-ignored labels (optional sanity metric)

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        # Move tensors to device (ignore non-tensors like strings/PIL)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["images"] = batch["images"].to(dtype=model_dtype) # [TODO]
        # print(f"in cot_vla_inference.py: batch['subgoal_images']: {batch['subgoal_images']}")
        # print(f"in cot_vla_inference.py: batch['labels'][0]: {batch['labels'][0]}")
        # batch = {
        #     "input_ids": input_ids,
        #     "labels": labels,
        #     "attention_mask": attention_mask,
        #     "images": images,
        #     "subgoal_images": valid_subgoal_images,
        # }

        # print("model dtype:", next(model.parameters()).dtype)
        # print("images dtype:", batch.get("images", None).dtype if batch.get("images", None) is not None else None)

        if mode == "Latent":
            out = model(
                input_ids=batch.get("input_ids", None),
                attention_mask=batch.get("attention_mask", None),
                labels=batch.get("labels", None),
                images=batch.get("images", None),   # <-- your forward takes `images`
                subgoal_images=batch.get("subgoal_images", None),
                num_extra_tokens=3, 
            )
        else:
            out = model(
                input_ids=batch.get("input_ids", None),
                attention_mask=batch.get("attention_mask", None),
                labels=batch.get("labels", None),
                images=batch.get("images", None),   # <-- your forward takes `images`
                subgoal_images=batch.get("subgoal_images", None),
                num_extra_tokens=2, 
            )

        # 1) loss
        print(out.image_loss)
        print(out.action_loss)

        if out.image_loss is not None:
            total_img_loss += float(out.image_loss.item())
        
        if out.action_loss is not None:
            total_act_loss += float(out.action_loss.item())
        
        n_batches += 1

        # 2) token prediction accuracy
        logits = out.logits                 # [B, S, V]
        preds = logits.argmax(dim=-1)       # [B, S]

        labels = batch["labels"]            # what dataset provides; shape depends on your collator
        input_ids = batch["input_ids"]   

        # print(f"logits: {logits}")
        # print(f"logits.shape: {logits.shape}")
        # print(f"preds = logits.argmax(dim=-1): {preds}")
        # print(f"preds.shape: {preds.shape}")
        # print(f"labels: {labels}")
        # print(f"labels.shape: {labels.shape}")
        # print(f"labels[0]: {labels[0]}")

        preds_shift = preds[:, :-1]   # [B, S-1]
        # tgts_shift = input_ids[:, 1:] # [B, S-1]
        
        tgts_shift = labels[:, 1:]
        # print(f"preds_shift: {preds_shift}")
        # print(f"labels_shift: {labels_shift}")
        
        B, S = tgts_shift.shape
        # print(f"B, S = {B}, {S}")
        preds_shift = preds_shift[:, -S:]
        # print(f"preds_shift.shape: {preds_shift.shape}")

        tgts_action_mask = torch.zeros_like(tgts_shift, dtype=torch.bool)
        for b in range(B):
            ids = labels[b]
            # print(f"(ids == ACTION_START_ID).nonzero(as_tuple=True): {(ids == ACTION_START_ID).nonzero(as_tuple=True)}")
            # print(f"(ids == ACTION_END_ID).nonzero(as_tuple=True): {(ids == ACTION_END_ID).nonzero(as_tuple=True)}")
            s = (ids == ACTION_START_ID).nonzero(as_tuple=True)[0][0]
            e = (ids == ACTION_END_ID).nonzero(as_tuple=True)[0][0]
            tgts_action_mask[b, s-1:e] = True
        
        tgts_action  = tgts_shift[tgts_action_mask]
        preds_action = preds_shift[tgts_action_mask]
        print(f"tgts_action: {tgts_action}")
        print(f"preds_action: {preds_action}")

        acc = (preds_action[1:-1] == tgts_action[1:-1]).float().mean()
        print(acc)
        accuracy.append(acc.item())

        start_acc = (
            (preds_action == ACTION_START_ID) &
            (tgts_action == ACTION_START_ID)
        ).float().sum() / (tgts_action == ACTION_START_ID).sum()
        
        accuracy_start.append(start_acc.item())

        end_acc = (
            (preds_action == ACTION_END_ID) &
            (tgts_action == ACTION_END_ID)
        ).float().sum() / (tgts_action == ACTION_END_ID).sum()

        accuracy_end.append(end_acc.item())
        
        # 3) MSE
        try:
            act = action_tokenizer.mixed_detokenize(preds_action)
            print("\n\nAction Tokenizer decode: ", act)
            gt_act = action_tokenizer.mixed_detokenize(tgts_action)
            print("\nGT Action: ", gt_act)

            se = (act[-2] - gt_act[-2]) ** 2
            # se = np.array(se)
            # print(f"se: {se}")
            mse_seq_per_dim = se.mean(axis=0)
            # print(f"mse_seq: {mse_seq}")
            mse_per_dim.append(mse_seq_per_dim)

            # se = (act[-2] - gt_act[-2]) ** 2 
            # print("Squared error: ", se)
            # mse_seq = se.mean(axis=0)
            # print("MSE along sequence dim: first dim: ", mse_seq)
            # mse_all = mse_seq.mean()
            # print("Total MSE: ",mse_all)
            # errors.append(mse_seq)
            valid += 1
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Invalid Generation")

    batch_mse_dim = np.stack(mse_per_dim, axis=0).mean(axis=0)

    metrics = {
        "avg_img_loss": total_img_loss / max(n_batches, 1),
        "avg_act_loss": total_act_loss / max(n_batches, 1),
        "action_token_acc": sum(accuracy) / len(accuracy),
        "action_start_acc": sum(accuracy_start) / len(accuracy_start),
        "action_end_acc": sum(accuracy_end) / len(accuracy_end),
        "avg_mse_per_dim": batch_mse_dim,
        "valid": valid,
        # "overall_token_acc_nonignored": (correct_all / total_all) if total_all else 0.0,
        # "boundary_token_acc": (correct_boundary / total_boundary) if total_boundary else 0.0,
        # "action_token_acc": (correct_action / total_action) if total_action else 0.0,
        # "total_boundary_tokens": float(total_boundary),
        # "total_action_tokens": float(total_action),
        "num_batches": float(n_batches),
    }
    return metrics



if __name__ == "__main__":
    @dataclass
    class EvalArguments:
        mode: str  # "VLA" | "COT" | "Latent"
        model_path: str

    parser = HfArgumentParser((EvalArguments, DataArguments))
    eval_args, data_args = parser.parse_args_into_dataclasses()
    # data_args = cast(Tuple[DataArguments], parser.parse_args_into_dataclasses())[0]
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True)
    # ### image/video understanding arguments
    # parser.add_argument("--image_path", type=str, default=None)
    # parser.add_argument("--video_path", type=str, default=None)
    # parser.add_argument("--query", type=str, default=None)
    # parser.add_argument("--temperature", type=float, default=0.9, help="The value of temperature for text generation.")
    # parser.add_argument("--top_p", type=float, default=0.6, help="The value of top-p for text generation.")
    # ### image and video generation arguments
    # parser.add_argument("--prompt", type=str, default=None)
    # parser.add_argument("--video_generation", type=bool, default=False)
    # parser.add_argument("--cfg", type=float, default=3.0, help="The value of the classifier free guidance for image generation.")
    # parser.add_argument("--save_path", type=str, default="generated_images/")
    # parser.add_argument("--generation_nums", type=int, default=1)
    # args = parser_0.parse_args()

    # if args.model_path is not None:
    #     model = vila_u.load(args.model_path)
    # else:
    #     raise ValueError("No model path provided!")

    # model_path = "checkpoints/latent_cotvla/checkpoint-560" # cotvla_vfix | latent_cotvla/checkpoint-560 | checkpoints/base_vla_v2/checkpoint-700
    model_path = eval_args.model_path
    # model_path = "vila-u-7b-256"
    model = vila_u.load(model_path)
    print("Loaded model from: ", model_path)

    vision_tower = model.get_vision_tower()
    print("Vision Tower:", vision_tower)
    print(vision_tower.config, vision_tower.is_loaded)
    # # img_proc = CLIPImageProcessor(
    # #         size={"height": 256, "width": 256}, 
    # #         crop_size={"height": 256, "width": 256}, 
    # #         image_mean=[0.5, 0.5, 0.5], 
    # #         image_std=[0.5, 0.5, 0.5]
    # #     )
    # image_tokens = 256

    # Manually write data args
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = True 
    data_args.mm_use_vi_start_end = True 
    print("Data args:", data_args)

    # Load data
    tokenizer = model.tokenizer
    action_tokenizer = ActionTokenizer(
        tokenizer=tokenizer
    )

    # vision_tower = model.get_vision_tower()
    # data_path = "data/rt1_100ss_20keps"
    data_path = "data/rt1_100ss_5n_7fg"
    print(f"Loading CoT Data From: {data_path}")

    eval_dataset = ShardedCoTVLADataset(
        data_dir=os.path.join(data_path, "eval"), # "eval" # Path to folder containing .npz shards
        tokenizer=tokenizer,
        data_args=data_args,
        # vision_tower=vision_tower,
        action_tokenizer=action_tokenizer,
        model_type=eval_args.mode,
    )

    print(f"Eval Dataset length: {len(eval_dataset)}")

    data_collator = CoTVLADataCollator(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=8, # 2
        collate_fn=data_collator
    )

    metrics = eval_teacher_forcing(model, dataloader, tokenizer, eval_args.mode, device="cuda", max_batches=50)
    print(metrics)