#!/usr/bin/env python3
"""
Systematic evaluation of KV cache compression methods.
Compares KNormCache and KVMergeCache against full caching baseline.
Now treats KL divergence and cosine distance **symmetrically** in metrics,
statistics, and plots.
"""

import argparse
import json
from pathlib import Path
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your cache implementations
from src.hf_cache import KNormCache
from src.kv_mean_merge import KVMergeCache


# ============================================================
# Data classes
# ============================================================

@dataclass
class CacheConfig:
    """Configuration for cache methods"""
    window_length: int = 64
    max_length: int = 128
    sim_threshold: float = 0.75  # For KVMergeCache


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    num_samples: int = 10
    max_tokens: int = 1024
    min_tokens: int = 512
    batch_size: int = 1  # Set to 1 if batching doesn't work with caches
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "./cache_eval_results"


@dataclass
class EvalMetrics:
    """Metrics for a single evaluation"""
    kl_divergence: List[float]
    cosine_distance: List[float]
    sequence_length: int

    # -------------------------------
    # KL divergence helpers
    # -------------------------------
    @property
    def avg_kl(self) -> float:
        return float(np.mean(self.kl_divergence))

    @property
    def kl_after_warmup(self) -> float:
        """KL divergence after cache warm‑up (default warm‑up = 500 tokens or ½ sequence)."""
        warmup = min(500, len(self.kl_divergence) // 2)
        return float(np.mean(self.kl_divergence[warmup:])) if len(self.kl_divergence) > warmup else self.avg_kl

    # -------------------------------
    # Cosine‑distance helpers (NEW)
    # -------------------------------
    @property
    def avg_cosine(self) -> float:
        return float(np.mean(self.cosine_distance))

    @property
    def cosine_after_warmup(self) -> float:
        """Cosine distance after cache warm‑up (mirrors `kl_after_warmup`)."""
        warmup = min(500, len(self.cosine_distance) // 2)
        return float(np.mean(self.cosine_distance[warmup:])) if len(self.cosine_distance) > warmup else self.avg_cosine


# ============================================================
# Evaluator
# ============================================================

class CacheEvaluator:
    """Evaluate cache compression methods using KL divergence *and* cosine distance."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Model & tokenizer
        print(f"Loading model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto" if config.device == "cuda" else None,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Data preparation
    # --------------------------------------------------

    def prepare_texts(self) -> List[str]:
        """Randomly sample sufficiently long passages from WikiText‑103."""
        print("Loading WikiText dataset…")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        paragraphs = [p for p in dataset["text"] if len(p.split()) > 15]

        texts = []
        for _ in range(self.config.num_samples):
            picked, cur_len = [], 0
            while cur_len < self.config.max_tokens * 4:  # rough char‑to‑token ratio
                paragraph = random.choice(paragraphs)
                picked.append(paragraph)
                cur_len += len(paragraph)
            texts.append("\n\n".join(picked))
        return texts

    def tokenize_and_truncate(self, text: str) -> Optional[torch.Tensor]:
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        if input_ids.size(1) > self.config.max_tokens:
            input_ids = input_ids[:, : self.config.max_tokens]
        return input_ids if input_ids.size(1) >= self.config.min_tokens else None

    # --------------------------------------------------
    # Metric computation
    # --------------------------------------------------

    @staticmethod
    def compute_metrics(logits_full: torch.Tensor, logits_cmp: torch.Tensor) -> Tuple[float, float]:
        """Return (kl_divergence, cosine_distance)."""
        logp_full = F.log_softmax(logits_full, dim=-1)
        p_cmp = F.softmax(logits_cmp, dim=-1)
        kl = F.kl_div(logp_full, p_cmp, reduction="batchmean").item()
        cos_dist = 1 - F.cosine_similarity(
            logits_full.flatten(1), logits_cmp.flatten(1), dim=1
        ).mean().item()
        return kl, cos_dist

    # --------------------------------------------------
    # Per‑sequence evaluation
    # --------------------------------------------------

    def evaluate_single_sequence(self, input_ids: torch.Tensor, cache_cfg: CacheConfig) -> Dict[str, EvalMetrics]:
        """Run one forward pass per token and record metrics for each cache."""
        n_steps = input_ids.size(1) - 1

        pkv_full = None
        pkv_knorm = KNormCache(window_length=cache_cfg.window_length, max_length=cache_cfg.max_length)
        pkv_kvmerge = KVMergeCache(
            window_length=cache_cfg.window_length,
            max_length=cache_cfg.max_length,
            sim_threshold=cache_cfg.sim_threshold,
        )

        metrics = {m: {"kl": [], "cos": []} for m in ["knorm", "kvmerge"]}

        self.model.eval()
        with torch.no_grad():
            for t in tqdm(range(n_steps), desc="Tokens", leave=False):
                x = input_ids[:, t : t + 1]

                # Full baseline
                out_full = self.model(x, past_key_values=pkv_full, use_cache=True, return_dict=True)
                pkv_full = out_full.past_key_values
                logits_full = out_full.logits[:, -1].float()

                # KNorm‑cache
                out_knorm = self.model(x, past_key_values=pkv_knorm, use_cache=True, return_dict=True)
                pkv_knorm = out_knorm.past_key_values
                logits_knorm = out_knorm.logits[:, -1].float()

                # KV‑merge cache
                out_kvmerge = self.model(x, past_key_values=pkv_kvmerge, use_cache=True, return_dict=True)
                pkv_kvmerge = out_kvmerge.past_key_values
                logits_kvmerge = out_kvmerge.logits[:, -1].float()

                # Metrics
                kl_k, cos_k = self.compute_metrics(logits_full, logits_knorm)
                kl_m, cos_m = self.compute_metrics(logits_full, logits_kvmerge)

                metrics["knorm"]["kl"].append(kl_k)
                metrics["knorm"]["cos"].append(cos_k)
                metrics["kvmerge"]["kl"].append(kl_m)
                metrics["kvmerge"]["cos"].append(cos_m)

        return {
            m: EvalMetrics(kl_divergence=metrics[m]["kl"], cosine_distance=metrics[m]["cos"], sequence_length=n_steps)
            for m in metrics
        }

    # --------------------------------------------------
    # High‑level evaluation loop
    # --------------------------------------------------

    def run_evaluation(self):
        texts = self.prepare_texts()
        print(f"Prepared {len(texts)} samples")

        cache_cfg = CacheConfig()
        all_results = {m: [] for m in ["knorm", "kvmerge"]}

        for idx, txt in enumerate(texts, 1):
            print(f"\nEvaluating sample {idx}/{len(texts)}")
            ids = self.tokenize_and_truncate(txt)
            if ids is None:
                print("  Skipped (too short)")
                continue
            print(f"  Sequence length: {ids.size(1)} tokens")

            seq_results = self.evaluate_single_sequence(ids, cache_cfg)
            for m in seq_results:
                all_results[m].append(seq_results[m])

        stats = self.compute_aggregate_stats(all_results)
        self.save_results(all_results, stats)
        self.generate_plots(all_results, stats)
        return stats

    # --------------------------------------------------
    # Statistics & I/O
    # --------------------------------------------------

    @staticmethod
    def _mean_std(values: List[float]) -> Tuple[float, float]:
        return float(np.mean(values)), float(np.std(values))

    def compute_aggregate_stats(self, results: Dict[str, List[EvalMetrics]]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for method, res_list in results.items():
            if not res_list:
                continue

            # Helper to collect arrays
            def collect(attr: str) -> List[float]:
                return [getattr(r, attr) for r in res_list]

            stats[method] = {
                # KL
                "avg_kl": self._mean_std(collect("avg_kl"))[0],
                "std_kl": self._mean_std(collect("avg_kl"))[1],
                "avg_kl_after_warmup": self._mean_std(collect("kl_after_warmup"))[0],
                "std_kl_after_warmup": self._mean_std(collect("kl_after_warmup"))[1],
                # Cosine
                "avg_cosine": self._mean_std(collect("avg_cosine"))[0],
                "std_cosine": self._mean_std(collect("avg_cosine"))[1],
                "avg_cosine_after_warmup": self._mean_std(collect("cosine_after_warmup"))[0],
                "std_cosine_after_warmup": self._mean_std(collect("cosine_after_warmup"))[1],
                # misc
                "num_samples": len(res_list),
            }
        return stats

    # ------------------------------------------------------------------
    # Save & report
    # ------------------------------------------------------------------

    def save_results(self, results: Dict[str, List[EvalMetrics]], stats: Dict):
        # Detailed per‑sequence metrics
        (self.output_dir / "detailed_results.json").write_text(
            json.dumps({m: [asdict(r) for r in rs] for m, rs in results.items()}, indent=2)
        )
        # Aggregate statistics
        (self.output_dir / "aggregate_stats.json").write_text(json.dumps(stats, indent=2))

        # Console summary ------------------------------------------------
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for m, s in stats.items():
            print(f"\n{m.upper()}: (n={s['num_samples']})")
            print(
                f"  KL divergence      : {s['avg_kl']:.6f} ± {s['std_kl']:.6f} | "
                f"warm‑up: {s['avg_kl_after_warmup']:.6f} ± {s['std_kl_after_warmup']:.6f}"
            )
            print(
                f"  Cosine distance    : {s['avg_cosine']:.6f} ± {s['std_cosine']:.6f} | "
                f"warm‑up: {s['avg_cosine_after_warmup']:.6f} ± {s['std_cosine_after_warmup']:.6f}"
            )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_metric_by_position(self, ax, results, metric_key: str, ylabel: str, log_y: bool = False):
        for method in ["knorm", "kvmerge"]:
            if not results[method]:
                continue
            min_len = min(r.sequence_length for r in results[method])
            avg_vals = [
                np.mean([getattr(r, metric_key)[pos] for r in results[method]])
                for pos in range(min_len)
            ]
            ax.plot(avg_vals, label=f"{method.upper()} (n={len(results[method])})")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Token position")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_distribution(self, ax, results, attr_full: str, attr_warm: str, title: str, ylab: str):
        for i, method in enumerate(["knorm", "kvmerge"]):
            if not results[method]:
                continue
            full_vals = [getattr(r, attr_full) for r in results[method]]
            warm_vals = [getattr(r, attr_warm) for r in results[method]]
            positions = [i * 2, i * 2 + 0.8]
            ax.boxplot(
                [full_vals, warm_vals],
                positions=positions,
                widths=0.6,
                labels=["Full", "After warmup"],
                patch_artist=True,
            )
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks([0.4, 2.4])
        ax.set_xticklabels(["KNorm", "KVMerge"])

    def generate_plots(self, results: Dict[str, List[EvalMetrics]], stats: Dict):
        # ------------------------------ KL divergence plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self._plot_metric_by_position(ax1, results, "kl_divergence", "Average KL divergence", log_y=True)
        ax1.set_title("Average KL Divergence vs Token Position")
        self._plot_distribution(
            ax2,
            results,
            "avg_kl",
            "kl_after_warmup",
            "Distribution of KL Divergence Values",
            "KL divergence",
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / "kl_plots.png", dpi=150)
        plt.close(fig)

        # ------------------------------ Cosine‑distance plots (mirrors KL)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self._plot_metric_by_position(ax1, results, "cosine_distance", "Average cosine distance")
        ax1.set_title("Average Cosine Distance vs Token Position")
        self._plot_distribution(
            ax2,
            results,
            "avg_cosine",
            "cosine_after_warmup",
            "Distribution of Cosine Distance Values",
            "Cosine distance",
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / "cosine_plots.png", dpi=150)
        plt.close(fig)


# ============================================================
# CLI wrapper
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate KV cache compression methods")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name or path")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of text samples to evaluate")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens per sample")
    parser.add_argument("--output-dir", type=str, default="./cache_eval_results", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = EvalConfig(
        model_name=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    evaluator = CacheEvaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()