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

    # -------- token IDs --------
    # Prefer constants if they exist in tokenizer; fallback to known ids if you hardcoded them.
    # In your code base you import ACTION_START/ACTION_END strings from vila_u.constants.
    # Usually tokenizer has them as added special tokens.
    try:
        ACTION_START_ID = tokenizer.convert_tokens_to_ids(ACTION_START)
        ACTION_END_ID   = tokenizer.convert_tokens_to_ids(ACTION_END)
    except Exception:
        # Fallback: if your teammate hardcoded action_start at 32135 etc.
        # Replace with your actual ids if needed.
        ACTION_START_ID = 32135
        ACTION_END_ID   = 32136

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
        print(f"in cot_vla_inference.py: batch['subgoal_images']: {batch['subgoal_images']}")
        print(f"in cot_vla_inference.py: batch['labels'][0]: {batch['labels'][0]}")
        # batch = {
        #     "input_ids": input_ids,
        #     "labels": labels,
        #     "attention_mask": attention_mask,
        #     "images": images,
        #     "subgoal_images": valid_subgoal_images,
        # }


        print("model dtype:", next(model.parameters()).dtype)
        print("images dtype:", batch.get("images", None).dtype if batch.get("images", None) is not None else None)

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
                # num_extra_tokens=2, 
            )

        # 1) loss
        print(out.image_loss)
        print(out.action_loss)

        if out.image_loss is not None:
            total_img_loss += float(out.image_loss.item())
        
        if out.action_loss is not None:
            total_act_loss += float(out.action_loss.item())
        
        n_batches += 1

        # 2) token predictions
        logits = out.logits                 # [B, S, V]
        preds = logits.argmax(dim=-1)       # [B, S]

        labels = batch["labels"]            # what dataset provides; shape depends on your collator
        # In your forward, you treat `new_labels[i]` as [S, something] and do label[:,0].
        # But externally, your dataset/collator might already be [B, S] (common).
        # Handle both cases:
        if labels.dim() == 3:
            labels_1d = labels[..., 0]      # [B, S]
        else:
            labels_1d = labels              # [B, S]

        # Align for next-token prediction (same shift convention as your forward)
        # model predicts token at t+1 using logits at t
        preds_shift  = preds[:, :-1]            # [B, S-1]
        labels_shift = labels_1d[:, 1:]         # [B, S-1]

        # ignore positions that are -100 (HF ignore_index)
        valid_mask = labels_shift.ne(-100)

        # # Optional: overall token accuracy on non-ignored labels (sanity check)
        # if valid_mask.any():
        #     correct_all += (preds_shift[valid_mask] == labels_shift[valid_mask]).sum().item()
        #     total_all += valid_mask.sum().item()

        # --- boundary token accuracy ---
        boundary_mask = valid_mask & (
            labels_shift.eq(ACTION_START_ID) | labels_shift.eq(ACTION_END_ID)
        )
        if boundary_mask.any():
            correct_boundary += (preds_shift[boundary_mask] == labels_shift[boundary_mask]).sum().item()
            total_boundary += boundary_mask.sum().item()

        # --- action token accuracy (between <action_start> and <action_end> in LABELS) ---
        # We locate action span based on label sequence (teacher-forcing ensures it exists if data is correct).
        B, S1 = labels_shift.shape
        for b in range(B):
            y = labels_shift[b]     # [S-1]
            p = preds_shift[b]      # [S-1]

            # find first start/end after shifting
            start_pos = (y == ACTION_START_ID).nonzero(as_tuple=False)
            end_pos   = (y == ACTION_END_ID).nonzero(as_tuple=False)

            if len(start_pos) == 0 or len(end_pos) == 0:
                continue

            s = int(start_pos[0].item())
            # choose the first end that occurs AFTER start
            e_candidates = end_pos[end_pos[:, 0] > s]
            if len(e_candidates) == 0:
                continue
            e = int(e_candidates[0].item())

            if e <= s + 1:
                continue  # nothing between

            # positions strictly between boundaries
            region = torch.arange(s + 1, e, device=y.device)
            # also exclude ignore-index locations (if any)
            region = region[y[region].ne(-100)]
            if region.numel() == 0:
                continue

            correct_action += (p[region] == y[region]).sum().item()
            total_action += region.numel()


    metrics = {
        "avg_img_loss": total_img_loss / max(n_batches, 1),
        "avg_act_loss": total_act_loss / max(n_batches, 1),
        # "overall_token_acc_nonignored": (correct_all / total_all) if total_all else 0.0,
        "boundary_token_acc": (correct_boundary / total_boundary) if total_boundary else 0.0,
        "action_token_acc": (correct_action / total_action) if total_action else 0.0,
        "total_boundary_tokens": float(total_boundary),
        "total_action_tokens": float(total_action),
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

    # model_path = "checkpoints/latent_cotvla/checkpoint-560" # cotvla_vfix | latent_cotvla/checkpoint-560
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
        data_dir=os.path.join(data_path, "eval"), # Path to folder containing .npz shards
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
        batch_size=2,
        collate_fn=data_collator
    )

    metrics = eval_teacher_forcing(model, dataloader, tokenizer, eval_args.mode, device="cuda", max_batches=50)
    print(metrics)

    print("Printing Train Dataset First Example:")
    print(eval_dataset[0])
    x = eval_dataset[0]
    pad_id = tokenizer.pad_token_id  # often 0 for LLaMA-style tokenizers
    clean_ids = [pad_id if x == -200 else x for x in x['input_ids']]
    print("Decoded Input: ")
    x_og_decode = tokenizer.decode(clean_ids, skip_special_tokens=False)
    print(x_og_decode)

    print("Action Tokenizer Decode: ")
    
    x_text = action_tokenizer.mixed_detokenize(clean_ids)
    print(x_text)
    print("Reference GT: ")
    print(eval_dataset.print_raw(0))

    # print("first dataset example:", x)
    
    first_raw = eval_dataset.get_raw(0)
    print("first dataset example:", first_raw)

    instruction = first_raw["curr_img_pil"]
    curr_img = first_raw["curr_img_pil"]
    # cur_img = Image.fromarray(curr_img_np)

    print("Image: ", curr_img)
    print("Instruction: ", instruction)
    # data_collator = CoTVLADataCollator(
    #     tokenizer=tokenizer,
    #     data_args=data_args,
    # )

    # dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=2,
    #     collate_fn=data_collator
    # )
    # # Get next batch
    # data_iterator = iter(dataloader)
    # next_batch = next(data_iterator)
    # print("Next Batch: ", next_batch)

    st = time.time()
    output_id = model.generate_cotvla([curr_img, instruction])[0].clone()

    et = time.time()
    print("model generate took: ", round(et-st, 2), "seconds")

    # Manually inject action start token at the front
    output_id[0] = 32135
    # outputs = model.generate(x["input_ids"], x["image"])
    print("output shape: ", output_id.shape)
    print("output from model: ", output_id)
    

    print("Original Tokenizer decode: ", tokenizer.decode(output_id, skip_special_tokens=False).strip())

    curr_img.save("testing_eval_img.jpg")
    act = action_tokenizer.mixed_detokenize(output_id)
    print("\n\nAction Tokenizer decode: ", act)
    # output_token_decode = tokenizer.decode(outputs, skip_special_tokens=False)

    # if args.query is not None:
    #     generation_config = model.default_generation_config
    #     generation_config.temperature = args.temperature
    #     generation_config.top_p = args.top_p
    #     if args.image_path is not None:
    #         image = vila_u.Image(args.image_path)
    #         response = model.generate_content([image, args.query])
    #         print("\033[1;32mResponse:\033[0m", response)
    #         exit()
    #     elif args.video_path is not None:
    #         video = vila_u.Video(args.video_path)
    #         response = model.generate_content([video, args.query])
    #         print("\033[1;32mResponse:\033[0m", response)
    #         exit()
    #     else:
    #         raise ValueError("No visual content input!")
    # elif args.prompt is not None:
    #     if args.video_generation:
    #         response = model.generate_video_content(args.prompt, args.cfg, args.generation_nums)
    #         save_video(response, args.save_path)
    #         exit()
    #     else:
    #         response = model.generate_image_content(args.prompt, args.cfg, args.generation_nums)
    #         save_image(response, args.save_path)
    #         exit()
    # else:
    #     raise ValueError("No query or prompt provided!")