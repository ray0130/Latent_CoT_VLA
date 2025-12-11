import argparse
import cv2
import numpy as np
import os
import vila_u


from vila_u.data.dataset import ShardedCoTVLADataset, CoTVLADataCollator
from vila_u.model.language_model.action_tokenizer_sepdim import ActionTokenizer
from transformers import HfArgumentParser, AutoConfig

from vila_u.train.args import TrainingArguments, ModelArguments, DataArguments
from typing import Dict, Tuple, cast

from transformers import CLIPImageProcessor

from torch.utils.data import DataLoader
from PIL import Image, ImageFile

import time

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
    parser = HfArgumentParser((DataArguments))
    data_args = cast(Tuple[DataArguments], parser.parse_args_into_dataclasses())[0]
    
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--model_path", type=str, required=True)
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
    # args = parser.parse_args()

    # if args.model_path is not None:
    #     model = vila_u.load(args.model_path)
    # else:
    #     raise ValueError("No model path provided!")

    model_path = "checkpoints/cot_vla_20keps"
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
    data_path = "data/rt1_100ss_20keps"
    print(f"Loading CoT Data From: {data_path}")

    eval_dataset = ShardedCoTVLADataset(
        data_dir=os.path.join(data_path, "eval"), # Path to folder containing .npz shards
        tokenizer=tokenizer,
        data_args=data_args,
        # vision_tower=vision_tower,
        action_tokenizer=action_tokenizer
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