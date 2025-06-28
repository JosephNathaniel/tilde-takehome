import json
from pathlib import Path
from basic_eval import CacheEvaluator, EvalMetrics, EvalConfig

def main():
    results_dir = Path("cache_eval_results")
    detailed_path = results_dir / "detailed_results.json"
    stats_path = results_dir / "aggregate_stats.json"

    # Load results
    with open(detailed_path, "r") as f:
        detailed = json.load(f)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Reconstruct EvalMetrics objects
    results = {}
    for method, method_results in detailed.items():
        results[method] = [
            EvalMetrics(
                kl_divergence=r["kl_divergence"],
                cosine_distance=r["cosine_distance"],
                sequence_length=r["sequence_length"],
            )
            for r in method_results
        ]

    # Dummy config for output_dir
    config = EvalConfig(output_dir=str(results_dir))
    evaluator = CacheEvaluator(config)
    evaluator.generate_plots(results, stats)

if __name__ == "__main__":
    main()