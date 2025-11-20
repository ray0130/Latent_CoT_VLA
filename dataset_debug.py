import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CoTDataset(Dataset):
    def __init__(self, shard_dir):
        """
        shard_dir: directory containing shard_*.npz files
        """
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if len(self.shard_paths) == 0:
            raise ValueError("No shards found in: " + shard_dir)

        print(f"Found {len(self.shard_paths)} shard files.")

        # Load all shards into a list of indices pointing to samples
        self.shards = []
        self.index_map = []   # (shard_index, sample_index_within_shard)

        for shard_i, path in enumerate(self.shard_paths):
            data = np.load(path, allow_pickle=True)
            self.shards.append(data)
            N = len(data["instruction"])
            for j in range(N):
                self.index_map.append((shard_i, j))

        print(f"Total samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_i, j = self.index_map[idx]
        data = self.shards[shard_i]

        return {
            "instruction": data["instruction"][j],
            "curr_visual_tokens": data["curr_visual_tokens"][j],            # raw image or tokens
            "subgoal_visual_tokens": data["subgoal_visual_tokens"][j],      # raw image or tokens
            "action_tokens": torch.tensor(data["action_tokens"][j], dtype=torch.long)
        }


def cot_collate_fn(batch):
    """
    batch: list of dicts returned by __getitem__
    Handles:
      - instructions (list of strings)
      - curr/subgoal: either raw images or visual tokens
      - actions: (m_action, 7) tensors
    """
    instructions = [item["instruction"] for item in batch]

    curr_batch = [item["curr_visual_tokens"] for item in batch]
    subgoal_batch = [item["subgoal_visual_tokens"] for item in batch]

    # Detect whether curr/subgoal are token sequences or raw images
    is_tokenized = isinstance(curr_batch[0], np.ndarray) and curr_batch[0].ndim == 1

    if is_tokenized:
        # Visual tokens -> stack into (B, N_tokens)
        curr_batch = torch.tensor(np.stack(curr_batch, axis=0), dtype=torch.long)
        subgoal_batch = torch.tensor(np.stack(subgoal_batch, axis=0), dtype=torch.long)
    else:
        # Raw images -> keep as Python list (for later transforms)
        # (H, W, C) as numpy arrays
        # E.g., you can convert them inside the model or preprocessing
        pass

    actions = torch.stack([item["action_tokens"] for item in batch], dim=0)   # (B, m, 7)

    return {
        "instruction": instructions,   # list of strings
        "curr_visual_tokens": curr_batch,
        "subgoal_visual_tokens": subgoal_batch,
        "action_tokens": actions
    }


from torch.utils.data import DataLoader

dataset = CoTDataset(OUTPUT_DIR)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=cot_collate_fn,
)


for batch in loader:
    print("batch size:", len(batch["instruction"]))
    print("Instructions:", batch["instruction"])
    print("batch['curr_visual_tokens']:", 
          batch["curr_visual_tokens"].shape if torch.is_tensor(batch["curr_visual_tokens"]) else len(batch["curr_visual_tokens"]))
    print(batch["curr_visual_tokens"])
    print("batch['subgoal_visual_tokens']:", 
          batch["subgoal_visual_tokens"].shape if torch.is_tensor(batch["subgoal_visual_tokens"]) else len(batch["subgoal_visual_tokens"]))
    print(batch["subgoal_visual_tokens"])
    print("batch['action_tokens']:", batch["action_tokens"].shape)
    print(batch["action_tokens"])
    break
