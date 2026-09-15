"""
Trains the exercise classifier, prints a full evaluation report, and
saves the trained model to models/exercise_classifier.joblib.

Run: python train_exercise_classifier.py

IMPORTANT: read the accuracy disclaimer this script prints before
trusting the numbers. See recognition/dataset_synthesis.py and
STATUS.md for the full explanation.
"""
import argparse
from recognition.classifier import train_classifier, save_model
from recognition.dataset_synthesis import EXERCISES
from evaluation.classification import classification_report


def print_confusion_matrix(report):
    labels = report.labels
    width = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l[:10]:>12}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:<{width}}" + "".join(f"{report.confusion_matrix[i, j]:>12}" for j in range(len(labels)))
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Train exercise classifier.")
    parser.add_argument("--use-qevd", "--qevd", action="store_true", help="Train and evaluate on QEVD-FIT-300k-Part-1 dataset subset.")
    parser.add_argument("--max-videos", type=int, default=50, help="Max QEVD videos when using --use-qevd (default: 50)")
    args = parser.parse_args()

    if args.use_qevd:
        import sys
        sys.argv = [sys.argv[0], "--max-videos", str(args.max_videos)]
        import train_on_qevd
        train_on_qevd.main()
        return

    print("Training exercise classifier on SYNTHETIC data...")
    print("(see recognition/dataset_synthesis.py for exactly what that means)\n")

    result = train_classifier(n_subjects=40, examples_per_subject_per_exercise=50, seed=0)
    report = result.report

    print(f"Training examples: {result.n_train}   Test examples: {result.n_test}")
    print("(train/test split by synthetic SUBJECT, not by individual example)\n")

    print("Confusion matrix (rows = true, columns = predicted):")
    print_confusion_matrix(report)
    print()

    print(f"Overall accuracy:  {report.accuracy * 100:.1f}%")
    print(f"Macro F1:          {report.macro_f1:.3f}\n")

    print(f"{'Exercise':<18}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    for label in EXERCISES:
        print(f"{label:<18}{report.precision[label]*100:>11.1f}%{report.recall[label]*100:>11.1f}%{report.f1[label]:>12.3f}")

    save_model(result.model, "models/exercise_classifier.joblib")
    print("\nModel saved to models/exercise_classifier.joblib")

    print("\n" + "=" * 70)
    print("ACCURACY DISCLAIMER — read this before using this number anywhere:")
    print("=" * 70)
    print(
        "This accuracy measures how well the classifier recovers the\n"
        "GEOMETRIC DEFINITIONS encoded in recognition/dataset_synthesis.py,\n"
        "on more synthetic examples generated the same way. Training and\n"
        "test data are BOTH synthetic. This is NOT a measurement of\n"
        "real-world accuracy on an actual camera feed with real tracking\n"
        "noise, real anatomical variation, or exercise variants this\n"
        "generator doesn't model. Real-world accuracy is UNKNOWN until this\n"
        "is evaluated against real recorded reps. See STATUS.md."
    )


if __name__ == "__main__":
    main()

