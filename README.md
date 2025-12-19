# Latent CoT VLA

Yueh An Liao, Ray Wen

Below are the steps to reproduce our findings.

## Environment setup

Our code works with VILA-U's original environment set up.

To set up, activate your conda environment first, and then run `./environment_setup.sh`

Or if you would like the script to create a conda environment for you, you can run `./environment_setup.sh <conda env name>`

**Note**: VILA-U's set up script replaces the packages transformer files with several of their own included in `vila_u/train/transformers_replace` so please make sure you are in a dedicated environment so these changes do not affect other projects.

## Data Retrieval

To obtain data, please run the script `dataset_5n_7fg.py`

you can set the `MAX_EPISODES` and `OUTPUT_DIR` to create a larger train dataset and a smaller eval dataset, stored in the same directory.

After running this file twice and generating one train and one eval dataset, you should have 2 directories containing shards of data, and they both share the same parent directory `data_path`

## Training Model

### Latent CoT VLA

Before you train your Latent CoT VLA model, make sure your `data_path` set in `vila_u/train/train.py` is set correctly to your parent directory of your train and eval created in the previous step.

You should also download VILA-U's pretrained model by following their steps listed in their repository and put it in this level: [download vila-u pretrained model](https://github.com/mit-han-lab/vila-u?tab=readme-ov-file#download-models)

To train the Latent CoT VLA model, you can run `./scripts/train/train_latent_cotvla.sh`, which is a modified version of VILA-U's original train script.

Within that script file, you will find different settings that you can change, such as `global_bs` and `acc_step` to control device and global batch sizes. Eval and save strategy, warm up ratio, learning rate etc. And this script by default will report to Weights & Biases.

Running the training script will save your models in the `checkpoints/model_name` directory.

## Evaluation

After you have finished training your model, below are the steps to run our token level evaluation and generation evaluation scripts.

### Token Level Accuracy Evaluation

To run token level accuracy evaluation, please first navigate into `cot_vla_inference.py` and change the `data_path` variable to point to your parent data directory (the one that has both `train` and `eval` in it).
You can also adjust the `max_batches` to None to evaluate the full `eval` dataset

Then you can run the evaluation script by:

```bash
python cot_vla_inference.py --mode Latent --model_path path/to/your/checkpoints/model_name
```

After this script finishes running, it should output the final result dictionary at the final line of output.

### Generation Evaluation

To run generation evaluation, The process is similar to the Token Level Accuracy. 
Please navigate into `mse_gen_inference.py` and change the `data_path` variable to point to your parent data directory (the one that has both `train` and `eval` in it).

Then you can run the evaluation script by:

```bash
python mse_gen_inference.py --mode Latent --model_path path/to/your/checkpoints/model_name
```

Similarly, this script's final output line should be the dictionary that holds the evaluation metrics.

# Other Models

To train and run evaluation on the two other models, please follow the steps listed below carefully:

## Train

To train either CoTVLA or VLA models, please make the following modifications to the script:

```python
# In vila_u/train/train.py
model_type = "COT" # Or "VLA" if you want to train vla
# You should also set the data directory correctly, same as above
data_path = "path/to/your/parent/data/directory"
```

You will also have to adjust code a little bit by removing the trainable subgoal head and subgoal token. Due to VILA-U's nature of hardcoding special token IDs, you will also need to slightly adjust the indexing of special tokens. Specifically, you need to make adjustments in the following files:

In `vila_u/model/vila_u_arch.py`:

```python
# Inside prepare_inputs_labels_for_multimodal()
img_start_token_id = self.llm.vocab_size - 4 - 2 # Change to -2 because we are no long adding subgoal_token into our vocabulary
# Inside initialize_vision_tokenizer()
# Remove SUBGOAL_TOKEN from BOTH the tokenizer.add_tokens function calls
num_new_tokens = tokenizer.add_tokens([ACTION_START, ACTION_END, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN], special_tokens=True)
num_new_tokens = tokenizer.add_tokens([ACTION_START, ACTION_END, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
```

In `vila_u/model/language_model/vila_u_llama.py`:

```python
# Inside __init__ of VILAULlamaModel
# Comment out the instantiation of 
self.subgoal_head

# Within forward()
# Set 
extra_tokens = 2 # Because we removed SUBGOAL_TOKEN earlier
```

After these changes, you should be able to train the model using the original training bash script `./scripts/train/train_latent_cotvla.sh` and you can adjust the `output_dir` in this bash script to align well with your designed model.

## Evaluation

To run evaluation scripts, please keep the changes made in the Train section as it is also needed for evaluation and generation.
<!-- you will also need to make a couple more adjustments in addition to the ones mentioned in the training section above. This is so that the model type is chosen correctly and the repair action sequence logic is working correctly. -->

### Token Level Accuracy Evaluation

For Token Level Accuracy Evaluation, you can follow the same steps listed in the Latent Model one, changing the `data_dir` and `max_batches` and running the script with: (`model_type = "COT" or "VLA"`)

```bash
python cot_vla_inference.py --mode model_type --model_path path/to/your/checkpoints/model_name
```

### Generation

For Generation Evaluation, you will also need to make a couple more adjustments in addition to the ones mentioned in the training section above. This is so that the model type is chosen correctly and the repair action sequence logic is working correctly.

Inside `mse_gen_inference.py`:

```python
action_start_id = 32004
action_end_id = 32005
start_position = 4 # 4 is for COT, change to 2 if you are running VLA
```

Inside `vila_u/model/vila_u_arch.py`:

```python
# In generate_vla()
length = 40 # 40 is for COT, change to 38 if you are running VLA
```

# Code Adaptation

This project is adapted and forked from VILA-U's open source project: [VILA-U](https://github.com/mit-han-lab/vila-u)

The action tokenizer is adapted from OpenVLA's open source Action Tokenizer: [Open VLA Action Tokenizer](https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py)
