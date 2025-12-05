# TODO
# directory, path
# !mkdir -p /openx
# !gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 ./openx/fractal20220817_data/0.1.0
# visual token, does the image need to be resized?? rt1 shape (256, 320, 3)


import os
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


DATASET_NAME = "fractal20220817_data"
SPLIT = "train"
N_SUBGOAL = 32 # curr: steps[t], subgoal: steps[t+N_SUBGOAL]
M_ACTION = 32 # action chunk: steps[t:t+M_ACTION]; shape (M_ACTION, 7)
MAX_EPISODES = 20000 # or set to small int while debugging
SHARD_SIZE = 100 # how many samples per saved shard

OUTPUT_DIR = "./data/rt1_100ss_20keps/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ACTION_BIN_EDGES_PATH = "./data/action_bin_edges.npy" # Path for bin edges for action discretization. shape (7, 257)


# [from Open_X_Embodiment_Datasets.ipynb]
def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}' # [TODO] /openx/{dataset_name}/{version}


def compute_action_bin_edges():
    """
    compute per-dimension 1st–99th percentile bin edges for 256-bin action discretization.
    """
    print("Computing action bin edges...")
    b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET_NAME))
    ds = b.as_dataset(split=SPLIT, read_config=tfds.ReadConfig(add_tfds_id=False))
    # ds = tfds.load(DATASET_NAME, split=SPLIT, shuffle_files=False)

    all_actions = []
    for epi_idx, episode in enumerate(tqdm(ds, desc="Episodes (bin stats)")):
        if MAX_EPISODES is not None and epi_idx >= MAX_EPISODES:
            break

        steps = list(tfds.as_numpy(episode["steps"]))
        for step in steps:
            action = step["action"]
            action_7d = np.concatenate([
                action["world_vector"],                # (3,)
                action["rotation_delta"],              # (3,)
                action["gripper_closedness_action"],   # (1,)
            ], axis=0).astype(np.float32)
            all_actions.append(action_7d)

    all_actions = np.stack(all_actions, axis=0)  # (N, 7)
    print("all_actions:", all_actions.shape)

    bin_edges = []
    for i in range(7):
        dim_vals = all_actions[:, i]
        q1 = np.percentile(dim_vals, 1.0)
        q99 = np.percentile(dim_vals, 99.0)
        edges = np.linspace(q1, q99, num=257, dtype=np.float32)  # 256 bins => 257 edges
        bin_edges.append(edges)

    bin_edges = np.stack(bin_edges, axis=0)  # (7, 257)
    print("bin_edges:", bin_edges.shape)
    np.save(ACTION_BIN_EDGES_PATH, bin_edges)
    print("Saved bin edges to", ACTION_BIN_EDGES_PATH)


# [TODO] vision tokenizer
def to_visual_token(image):
    """
    Placeholder for (our) VILA-U RQ-VAE tokenizer.
    image: HxWxC numpy array (uint8)
    Returns: 1D array of visual token IDs, e.g. length 1024.
    """
    # TODO: plug in actual tokenizer
    # tokens = vila_tokenizer.encode(image)  # -> np.array of ints
    # For now, just return dummy zeros for debugging. Assume 16x16x4 = 1024 tokens
    h, w, c = image.shape
    return np.zeros((1024,), dtype=np.int32)


def discretize_action_7d(action_7d, bin_edges):
    """
    action_7d: (7,) continuous
    bin_edges: (7, 257) array of bin edges for np.digitize, per dimension
    Returns: (7,) integer tokens in [0, 255]
    """
    tokens = []
    for i in range(7):
        idx = np.digitize(action_7d[i], bin_edges[i]) - 1 # np.digitize returns 1..len(edges), we shift to 0..255
        idx = np.clip(idx, 0, 255) # need this cuz digitize() can return out-of-range indices when x falls outside the edges
        tokens.append(idx)
    return np.array(tokens, dtype=np.int32)


def build_trajectories():
    print("Loading Raw Actions. Not Binned")
    # print("Loading action bin edges from", ACTION_BIN_EDGES_PATH)
    # bin_edges = np.load(ACTION_BIN_EDGES_PATH)  # (7, 257)

    b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET_NAME))
    ds = b.as_dataset(split=SPLIT, read_config=tfds.ReadConfig(add_tfds_id=False))
    # ds = tfds.load(DATASET_NAME, split=SPLIT, shuffle_files=False)

    shard_idx = 0
    buffer = {
        "instruction": [],
        "curr_img": [],
        "subgoal_img": [],
        "action_vec": [], # shape (M_ACTION, 7)
    }

    def flush_buffer():
        nonlocal shard_idx, buffer
        if len(buffer["instruction"]) == 0:
            return
        out_path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:04d}.npz")
        np.savez_compressed(
            out_path,
            instruction=np.array(buffer["instruction"], dtype=object),
            curr_img=np.stack(buffer["curr_img"], axis=0),
            subgoal_img=np.stack(buffer["subgoal_img"], axis=0),
            action_vec=np.stack(buffer["action_vec"], axis=0),
        )
        print(f"Saved shard {shard_idx} with {len(buffer['instruction'])} samples -> {out_path}")
        shard_idx += 1
        buffer = {
            "instruction": [],
            "curr_img": [],
            "subgoal_img": [],
            "action_vec": [],
        }

    for epi_idx, episode in enumerate(tqdm(ds, desc="Episodes (windowing)")):
        if MAX_EPISODES is not None and epi_idx >= MAX_EPISODES:
            break

        # Episode steps as numpy
        steps = list(tfds.as_numpy(episode["steps"]))
        T = len(steps)
        if T == 0:
            continue
        
        # === Instruction ===
        instr_bytes = steps[0]["observation"]["natural_language_instruction"]
        # [TODO] ??Sometimes this is a scalar bytes or array of bytes
        if isinstance(instr_bytes, (np.ndarray, list)):
            instr = instr_bytes[0].decode("utf-8")
        else:
            instr = instr_bytes.decode("utf-8")

        # Sliding windows over steps[0] ~ steps[T-1]
        for t in range(T):
            if t + N_SUBGOAL > T-1: # curr: steps[t], subgoal: steps[t+N_SUBGOAL]
                break
            if t + M_ACTION > T-1: # action chunk: steps[t:t+M_ACTION]
                break

            curr_step = steps[t]
            subgoal_step = steps[t + N_SUBGOAL]

            # === Visual tokens ===
            curr_img = curr_step["observation"]["image"] # (256, 320, 3)
            subgoal_img = subgoal_step["observation"]["image"]
            # curr_vis_tokens = to_visual_token(curr_img)
            # subgoal_vis_tokens = to_visual_token(subgoal_img)

            # === Action tokens: (M_ACTION, 7) ===
            act_tokens = []
            for k in range(M_ACTION):
                action_7d = np.concatenate([
                    steps[t + k]["action"]["world_vector"],                # (3,)
                    steps[t + k]["action"]["rotation_delta"],              # (3,)
                    steps[t + k]["action"]["gripper_closedness_action"],   # (1,)
                ], axis=0).astype(np.float32)
                # tok7 = discretize_action_7d(action_7d, bin_edges)
                # act_tokens.append(tok7)
                act_tokens.append(action_7d)
            act_tokens = np.stack(act_tokens, axis=0)  # (M_ACTION, 7)

            # Append to buffer
            buffer["instruction"].append(instr)
            buffer["curr_img"].append(curr_img)
            buffer["subgoal_img"].append(subgoal_img)
            buffer["action_vec"].append(act_tokens)

            # Flush if shard full
            if len(buffer["instruction"]) >= SHARD_SIZE:
                print("flushing shard")
                flush_buffer()
                print("finished: flushing shard")

    # Final flush
    flush_buffer()
    print("Done!")


if __name__ == "__main__":
    # 1) compute bin edges for action discretization
    # if not os.path.exists(ACTION_BIN_EDGES_PATH):
    #     compute_action_bin_edges()
    # else:
    #     print("Bin edges already exist, skipping computation.")
    print("Skipping computing Bin since we save raw actions")
    # 2) trajectory + tokenize + save shards
    build_trajectories()