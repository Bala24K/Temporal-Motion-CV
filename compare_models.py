"""
Runs the multi-model comparison (Random Forest / XGBoost / MLP / temporal
1D-CNN) on the structured skeleton representation and prints a comparison
table.

Run: python compare_models.py

IMPORTANT: read the disclaimer this script prints. All models are trained
and evaluated on SYNTHETIC data — see recognition/pose_sequence_synthesis.py
and STATUS.md for exactly what that does and doesn't validate.
"""
import argparse
import time
from recognition.model_comparison import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison.")
    parser.add_argument("--use-qevd", "--qevd", action="store_true", help="Compare models on QEVD-FIT-300k-Part-1 dataset subset.")
    parser.add_argument("--max-videos", type=int, default=50, help="Max QEVD videos when using --use-qevd (default: 50)")
    args = parser.parse_args()

    if args.use_qevd:
        import sys
        sys.argv = [sys.argv[0], "--max-videos", str(args.max_videos)]
        import train_on_qevd
        train_on_qevd.main()
        return

    print("Running multi-model comparison on the structured skeleton representation...")
    print("(SYNTHETIC data — see recognition/pose_sequence_synthesis.py)\n")

    start = time.time()
    results = run_comparison(n_subjects=35, windows_per_subject_per_exercise=20,
                              n_frames=16, seed=0, cnn_epochs=50)
    elapsed = time.time() - start

    print(f"{'Model':<16}{'Accuracy':>12}{'Macro F1':>12}   n_train / n_test")
    print("-" * 60)
    for name, result in results.items():
        r = result.report
        print(f"{name:<16}{r.accuracy*100:>11.1f}%{r.macro_f1:>12.3f}   {result.n_train} / {result.n_test}")

    print(f"\n(comparison took {elapsed:.1f}s)")

    print("\nPer-exercise F1, by model:")
    exercises = list(next(iter(results.values())).report.f1.keys())
    header = f"{'Exercise':<18}" + "".join(f"{name:>16}" for name in results.keys())
    print(header)
    for ex in exercises:
        row = f"{ex:<18}" + "".join(f"{results[name].report.f1[ex]:>16.3f}" for name in results.keys())
        print(row)

    print("\n" + "=" * 70)
    print("DISCLAIMER — same as train_exercise_classifier.py's, and it matters just as much here:")
    print("=" * 70)
    print(
        "This compares how well each model recovers the GEOMETRIC\n"
        "DEFINITIONS encoded in recognition/pose_sequence_synthesis.py, on\n"
        "more synthetic windows generated the same way. It answers 'which\n"
        "model learns best from this representation, on this synthetic\n"
        "data' — NOT 'which model will perform best on real video.' See\n"
        "STATUS.md."
    )


if __name__ == "__main__":
    main()
