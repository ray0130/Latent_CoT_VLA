"""


Adapted from OpenVLA: https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py
"""
import numpy as np
from typing import Union, List, Sequence
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_actions: Union[float, Sequence[float]] = -1.0,
        max_actions: Union[float, Sequence[float]] = 1.0,
        n_dims: int = 7,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        Assumes a BPE style tokenizer akin to LlamaTokenizer, where the least used tokens
        appear at the end of the vocabulary.

        - tokenizer: base LLM/VLM tokenizer to extend
        - bins: number of bins per dimension (number of bin edges is `bins`)
        - min_actions: scalar or sequence of length n_dims giving per-dimension minimum
        - max_actions: scalar or sequence of length n_dims giving per-dimension maximum
        - n_dims: number of action dimensions (e.g. 7)
        """
        self.tokenizer = tokenizer
        self.n_bins = int(bins)
        self.n_dims = int(n_dims)

        # Convert min/max into per-dimension arrays
        # min_arr = np.asarray(min_actions, dtype=float)
        # max_arr = np.asarray(max_actions, dtype=float)
        min_arr = np.array([-0.224535, -0.148200, -0.231590, -0.351799, -0.419301, -0.436435, -1.0], dtype=float)
        max_arr = np.array([ 0.178247,  0.149384,  0.218423,  0.589267,  0.352727,  0.447967,  1.0], dtype=float)

        if min_arr.shape == ():
            min_arr = np.full(self.n_dims, float(min_arr), dtype=float)
        if max_arr.shape == ():
            max_arr = np.full(self.n_dims, float(max_arr), dtype=float)

        if min_arr.shape != (self.n_dims,) or max_arr.shape != (self.n_dims,):
            raise ValueError(
                f"min_actions and max_actions must be scalar or length {self.n_dims}, "
                f"got shapes {min_arr.shape} and {max_arr.shape}"
            )

        self.min_actions = min_arr
        self.max_actions = max_arr
        print("Min Actions: ", self.min_actions)
        print("Max Actions: ", self.max_actions)
        # Create per-dimension uniform bins
        # Shape: (n_dims, n_bins)
        self.bins = np.linspace(
            self.min_actions[:, None],
            self.max_actions[:, None],
            self.n_bins,
            axis=-1,
        ).squeeze(1)

        # Bin centers per dimension
        # Shape: (n_dims, n_bins - 1)
        self.bin_centers = (self.bins[:, :-1] + self.bins[:, 1:]) / 2.0

        # Keep the same contract for where action tokens live in the vocab
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def _discretize(self, action: np.ndarray) -> np.ndarray:
        """
        Internal helper:
        - action: shape (..., n_dims)
        - returns: integer bin indices with shape (..., n_dims), in [1, n_bins], like np.digitize
        """
        action = np.asarray(action, dtype=float)

        if action.shape[-1] != self.n_dims:
            raise ValueError(
                f"Expected last dimension size {self.n_dims} for actions, got {action.shape[-1]}"
            )

        # Clip per dimension using broadcasting
        # min_actions/max_actions broadcast over leading dimensions
        clipped = np.clip(action, self.min_actions, self.max_actions)

        # Flatten leading dims for easier looping over dimensions
        orig_shape = clipped.shape
        flat = clipped.reshape(-1, self.n_dims)  # (N, n_dims)

        discretized_flat = np.empty_like(flat, dtype=np.int64)

        # Per-dimension digitization
        for d in range(self.n_dims):
            discretized_flat[:, d] = np.digitize(flat[:, d], self.bins[d])

        discretized = discretized_flat.reshape(orig_shape).astype(np.int64)
        return discretized

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        Clip and bin actions to the last `n_bins` tokens of the vocabulary.

        - action: shape (n_dims,) or (T, n_dims) or (B, T, n_dims)
        - returns: string if single action, or list of strings otherwise
        """
        discretized_action = self._discretize(action)

        # Map bin indices [1, n_bins] to token ids at the end of the vocab
        token_ids = self.tokenizer.vocab_size - discretized_action

        if token_ids.ndim == 1:
            # Single action vector -> single sequence string
            return self.tokenizer.decode(token_ids.tolist())
        else:
            # Treat leading dims as sequence dims, last dim is tokens per step
            # For batch_decode, flatten leading dims to a batch dimension of sequences
            flat = token_ids.reshape(-1, token_ids.shape[-1])  # (N, n_dims)
            decoded = self.tokenizer.batch_decode(flat.tolist())
            return decoded

    def encode_actions(self, action: np.ndarray) -> np.ndarray:
        """
        Return integer token ids for the given continuous action array.

        - action: shape (n_dims,) or (T, n_dims) or (B, T, n_dims)
        - returns: np.ndarray of int64 token ids with same shape as action
        """
        discretized_action = self._discretize(action)
        token_ids = self.tokenizer.vocab_size - discretized_action
        return token_ids.astype(np.int64)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Return continuous actions for discrete action token IDs.

        - action_token_ids: shape (n_dims,) or (T, n_dims) or (B, T, n_dims)
        - returns: continuous actions with same shape as action_token_ids
        """
        action_token_ids = np.asarray(action_token_ids, dtype=np.int64)

        if action_token_ids.shape[-1] != self.n_dims:
            raise ValueError(
                f"Expected last dimension size {self.n_dims} for action tokens, got {action_token_ids.shape[-1]}"
            )

        # Recover digitized bin indices
        discretized_actions = self.tokenizer.vocab_size - action_token_ids

        # Map to indices for bin_centers: [0, n_bins - 2]
        idx = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[1] - 1)

        # Flatten leading dims for per-dimension indexing
        orig_shape = idx.shape
        flat_idx = idx.reshape(-1, self.n_dims)  # (N, n_dims)
        flat_actions = np.empty_like(flat_idx, dtype=np.float32)

        for d in range(self.n_dims):
            flat_actions[:, d] = self.bin_centers[d][flat_idx[:, d]]

        actions = flat_actions.reshape(orig_shape).astype(np.float32)
        return actions

    def mixed_detokenize(self, input_ids):
        """
        Detokenizes a sequence of ids that contains:
        text, <action_start>, action tokens (flattened), <action_end>, text.
        
        Returns:
            text_before: str
            actions: np.ndarray of shape (T, self.n_dims)
            text_after: str
        """

        # Convert to list if it is a tensor
        if hasattr(input_ids, "tolist"):
            ids = input_ids.tolist()
        else:
            ids = list(input_ids)

        # Convert tokens to ids for the markers
        start_id = self.tokenizer.convert_tokens_to_ids("<action_start>")
        end_id = self.tokenizer.convert_tokens_to_ids("<action_end>")

        # Locate start and end
        if start_id not in ids or end_id not in ids:
            raise ValueError("Input sequence must contain <action_start> and <action_end> tokens.")

        s = ids.index(start_id)
        e = ids.index(end_id)

        # Split into three parts
        text_before_ids = ids[:s]
        action_ids_flat = ids[s + 1 : e]
        text_after_ids = ids[e + 1 :]

        # Detokenize the text parts
        text_before = self.tokenizer.decode(text_before_ids, skip_special_tokens=False)
        text_after = self.tokenizer.decode(text_after_ids, skip_special_tokens=False)

        # Remove any placeholders such as -200 if they appear
        clean_action_ids = [i for i in action_ids_flat if i >= 0]

        # Make sure action ids length is divisible by the action dimension
        if len(clean_action_ids) % self.n_dims != 0:
            raise ValueError(
                f"Action token block length {len(clean_action_ids)}"
                f" is not divisible by action dimension {self.n_dims}."
            )

        # Reshape to (T, self.n_dims)
        action_token_matrix = np.array(clean_action_ids, dtype=np.int64).reshape(-1, self.n_dims)

        # Decode to continuous actions
        actions = self.decode_token_ids_to_actions(action_token_matrix)

        return text_before, actions, text_after

    @property
    def vocab_size(self) -> int:
        # Number of action bins (per dimension).
        return self.n_bins
