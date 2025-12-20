import argparse
import cv2
import numpy as np
import os
import vila_u
import torch


from vila_u.data.dataset import ShardedCoTVLADataset, CoTVLADataCollator
from vila_u.model.language_model.action_tokenizer_sepdim import ActionTokenizer
from transformers import HfArgumentParser, AutoConfig

from vila_u.train.args import TrainingArguments, ModelArguments, DataArguments
from typing import Dict, Tuple, cast

from transformers import CLIPImageProcessor

from torch.utils.data import DataLoader
from PIL import Image, ImageFile

import time
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


if __name__ == "__main__":
    @dataclass
    class EvalArguments:
        mode: str  # "VLA" | "COT" | "Latent"
        model_path: str
        N: int # number of samples

    parser = HfArgumentParser((EvalArguments, DataArguments))
    
    eval_args, data_args = parser.parse_args_into_dataclasses()
    
    
    model_path = eval_args.model_path
    model = vila_u.load(model_path)
    print("Loaded model from: ", model_path)

    vision_tower = model.get_vision_tower()
    print("Vision Tower:", vision_tower)
    print(vision_tower.config, vision_tower.is_loaded)
    

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
    # data_path = "data/rt1_100ss_20keps"
    data_path = "./data/rt1_100ss_5n_7fg"
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
    sum_error = 0
    errors = []
    valid = 0
    repaired = 0
    N = eval_args.N
    st_all = time.time()
    for i in range(N):
        first_raw = eval_dataset.get_raw(i)
        print("first dataset example:", first_raw)

        instruction = first_raw["instr"]
        curr_img = first_raw["curr_img_pil"]
        act_vec = first_raw["action_seq"]
        # cur_img = Image.fromarray(curr_img_np)

        print("Image: ", curr_img)
        print("Instruction: ", instruction)
        
        
        st = time.time()
        output_id = model.generate_vla([curr_img, instruction])[0].clone() # ?? generate_vla

        et = time.time()
        print("model generate took: ", round(et-st, 2), "seconds")

        # output_id[0] = 
        # Manually inject action start token at the front and end action at the end
        # output_id[0] = 32004
        # output_id[-1] = 32006

        # for COT VLA, start (32004) position 4, end (32005) position -1, total length 5 * 7 + 5
        # Base VLA, start (32004) position 2, end (32005) position -1, total length 5 * 7 + 3
        # Latent VLA, start (32005) position 3, end (32006) position -1, total length 5 * 7 + 4
        #
        action_start_id = 32005
        action_end_id = 32006
        start_position = 3
        repair_flag = False
        if output_id[start_position] != action_start_id:
            output_id[start_position] = action_start_id
            repair_flag = True
        if output_id[-1] != action_end_id:
            output_id[-1] = action_end_id
            repair_flag = True
        
        if repair_flag:
            print("Repaired Example")
            repaired += 1
        # output_id = torch.cat([output_id, torch.tensor([32005], dtype=output_id.dtype, device=output_id.device)], dim=0)
        # outputs = model.generate(x["input_ids"], x["image"])
        pad_id = tokenizer.pad_token_id
        print("pad id: ", pad_id)
        print("output shape: ", output_id.shape)
        print("output from model: ", output_id)
        

        print("Original Tokenizer decode: ", tokenizer.decode(output_id, skip_special_tokens=False).strip())
        try:
            # curr_img.save("testing_eval_img_overfit.jpg")
            act = action_tokenizer.mixed_detokenize(output_id)
            print("\n\nAction Tokenizer decode: ", act)
            gt_act = action_tokenizer.mixed_detokenize(act_vec)
            print("\nGT Action: ", gt_act)

            se = (act[-2] - gt_act[-2]) ** 2 
            print("Squared error: ", se)
            mse_seq = se.mean(axis=0)
            print("MSE along sequence dim: first dim: ", mse_seq)
            mse_all = mse_seq.mean()
            print("Total MSE: ",mse_all)
            errors.append(mse_seq)
            valid += 1
        except Exception as e:
            print("Invalid Generation")

    errors = np.stack(errors, axis=0)
    errors_seq = errors.sum(axis=0)
    print(f"Number of Repaired generations: {repaired}, / {N}")
    print(f"Number of valid generations: {valid}, / {N}")
    print("Mean error across batch: ", errors_seq / valid)
    errors_total = errors_seq.sum()
    print("Total: ", errors_total)

    print("Entire generate took: ", round(time.time()-st_all, 2), "seconds")
    # output_token_decode = tokenizer.decode(outputs, skip_special_tokens=False)
