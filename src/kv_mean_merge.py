from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache


class KVMergeCache(Cache):
    """Drop-in KV-cache implementing a **simplified KVMerger** (Wang et al., 2024).

    Instrumented so you can confirm merge activity.

    ### User knobs
    * **max_length** – total cache budget (old+recent)
    * **window_length** – newest tokens kept verbatim (no merges)
    * **sim_threshold** – cosine‑sim threshold for merging consecutive tokens
    * **verbose** – print per‑update diagnostics if *True*
    """

    # Required for 🤗 static‑cache forward path
    is_sliding = False

    # ---------------------------------------------------------------------
    # Construction & bookkeeping
    # ---------------------------------------------------------------------

    def __init__(
        self,
        max_length: int,
        window_length: int,
        sim_threshold: float = 0.75,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if window_length >= max_length:
            raise ValueError("window_length must be < max_length")

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.max_length = max_length
        self.window_length = window_length
        self.sim_threshold = sim_threshold
        self.verbose = verbose

        # runtime counters  -------------------------------------------------
        self._n_tokens_in = 0
        self._n_tokens_after_merge = 0
        self._n_merged_pairs = 0
        self._n_fallback_pruned = 0
        self._max_live_len = 0

    # ---------------------------------------------------------------------
    # Public inspector
    # ---------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return dictionary with runtime statistics."""
        return {
            "n_tokens_in": int(self._n_tokens_in),
            "n_tokens_after_merge": int(self._n_tokens_after_merge),
            "n_merged_pairs": int(self._n_merged_pairs),
            "n_fallback_pruned": int(self._n_fallback_pruned),
            "max_live_len": int(self._max_live_len),
        }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _cluster_merge(
        keys: torch.Tensor,
        values: torch.Tensor,
        thr: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Collapse each *contiguous* cluster of highly-similar tokens into a
        single key/value that is the **arithmetic mean of the whole cluster**.

        A cluster is a maximal run of tokens whose *adjacent* cosine similarity
        exceeds `thr` **in every head**.

        Returns
        -------
        merged_k : Tensor  (bsz, n_heads, ≤seq_len, dim)
        merged_v : Tensor  (bsz, n_heads, ≤seq_len, dim)
        n_pairs_merged : int   # how many token-pairs were merged away
        """
        assert keys.shape == values.shape, "Keys and values must have the same shape"
        bsz, n_heads, seq_len, dim = keys.shape
        if seq_len <= 1:
            return keys, values, 0

        # cosine similarity of every consecutive pair
        #   cos_sim shape: (bsz, n_heads, seq_len-1)
        cos_sim = F.cosine_similarity(
            keys[..., 1:, :], keys[..., :-1, :], dim=-1
        )

        # True where ALL heads are above threshold  → tokens belong together
        #   similar shape: (bsz, seq_len-1)
        similar = (cos_sim > thr).all(dim=1) # NOTE: doing a .all() across the head_dim. Will be false if ANY head is below threshold

        merged_k_batches: List[torch.Tensor] = []
        merged_v_batches: List[torch.Tensor] = []
        n_pairs_merged = 0

        # loop over batch dim
        for b in range(bsz):
            clusters_k: List[torch.Tensor] = []
            clusters_v: List[torch.Tensor] = []
            start = 0

            # sweep from left to right; a False → boundary between clusters
            for i in range(seq_len - 1):
                if not similar[b, i]:
                    k_slice = keys[b, :, start : i + 1, : ] # NOTE: slices across ALL heads in this layer
                    v_slice = values[b, :, start : i + 1, :]
                    L = k_slice.shape[-2] # number of tokens in this cluster
                    clusters_k.append(k_slice.mean(dim=-2, keepdim=True))
                    clusters_v.append(v_slice.mean(dim=-2, keepdim=True))
                    n_pairs_merged += L - 1
                    start = i + 1

            # flush final cluster
            k_slice = keys[b, :, start:, :]
            v_slice = values[b, :, start:, :]
            L = k_slice.shape[-2]
            clusters_k.append(k_slice.mean(dim=-2, keepdim=True))
            clusters_v.append(v_slice.mean(dim=-2, keepdim=True))
            n_pairs_merged += L - 1

            # (n_heads, n_clusters, dim)  → re-add batch dim
            # This is still inside the batch loop!
            merged_k_batches.append(torch.cat(clusters_k, dim=-2).unsqueeze(0))
            merged_v_batches.append(torch.cat(clusters_v, dim=-2).unsqueeze(0))

        # this won't work for multiple batches in general?
        merged_k = torch.cat(merged_k_batches, dim=0)
        merged_v = torch.cat(merged_v_batches, dim=0)
        return merged_k, merged_v, n_pairs_merged    

    # ---------------------------------------------------------------------
    # 🤗 Cache interface
    # ---------------------------------------------------------------------

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:  # noqa: D401
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        return self.window_length

    # ---------------------------------------------------------------------
    # Main update hook called by model forward
    # ---------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._n_tokens_in += key_states.shape[-2]

        # Append new KV to layer‑specific buffers --------------------------
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        seq_len = k.shape[-2]
        self._max_live_len = max(self._max_live_len, seq_len)

        # Exit early if still under budget ---------------------------------
        if seq_len <= self.max_length:
            self._n_tokens_after_merge = seq_len
            return k, v

        # Split into [compressible | recent_window]
        split = seq_len - self.window_length
        comp_k, recent_k = k[..., :split, :], k[..., split:, :]
        comp_v, recent_v = v[..., :split, :], v[..., split:, :]

        # Merge similar consecutive tokens --------------------------------
        assert comp_k.shape == comp_v.shape, "Keys and values must have the same shape"
        comp_k, comp_v, n_merged = self._cluster_merge(comp_k, comp_v, self.sim_threshold)
        self._n_merged_pairs += n_merged

        # Top‑K fallback if still over budget ------------------------------
        target_len = self.max_length - self.window_length
        excess = comp_k.shape[-2] - target_len
        if excess > 0:
            self._n_fallback_pruned += excess
            if self.verbose:
                print(f"[KVMergeCache] fallback prune {excess} tokens → Top‑norm")
            norms = comp_k.norm(dim=-1)
            idx = (-norms).topk(target_len, dim=-1).indices.sort().values
            idx_exp = idx[..., None].repeat_interleave(comp_k.shape[-1], dim=-1)
            comp_k = torch.gather(comp_k, -2, idx_exp)
            comp_v = torch.gather(comp_v, -2, idx_exp)

        # Re‑assemble full cache -------------------------------------------
        self.key_cache[layer_idx] = torch.cat((comp_k, recent_k), dim=-2)
        self.value_cache[layer_idx] = torch.cat((comp_v, recent_v), dim=-2)
        self._n_tokens_after_merge = self.key_cache[layer_idx].shape[-2]

        if self.verbose:
            print(
                f"[KVMergeCache] layer {layer_idx} | merges={n_merged} | len={self.key_cache[layer_idx].shape[-2]}"
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
