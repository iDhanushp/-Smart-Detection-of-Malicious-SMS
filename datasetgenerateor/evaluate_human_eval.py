"""Evaluate deployed TFLite model on a human-labeled gold CSV.

Required columns in input CSV:
- sender
- body
- corrected_label  (legit|spam|fraud)

Optional columns are preserved in error report.

Usage:
  python evaluate_human_eval.py --gold human_eval_gold.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def load_feature_extractor(train_script_path: str):
    src = open(train_script_path, encoding="utf-8").read()
    markers = [
        "# ─── load & label data",
        "# --- load & label data",
        "# load & label data",
    ]

    cut = None
    for marker in markers:
        idx = src.find(marker)
        if idx != -1:
            cut = idx
            break
    if cut is None:
        raise RuntimeError("Could not find load marker in train_real_model.py")

    scope: dict[str, object] = {
        "__file__": train_script_path,
        "__name__": "__feature_loader__",
    }
    exec(compile(src[:cut], train_script_path, "exec"), scope)
    if "extract_features" not in scope:
        raise RuntimeError("extract_features() not found")
    if "label_message" not in scope:
        raise RuntimeError("label_message() not found")
    return scope["extract_features"], scope["label_message"]


def accuracy_ci(acc: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    se = math.sqrt(max(acc * (1.0 - acc), 0.0) / n)
    lo = max(0.0, acc - z * se)
    hi = min(1.0, acc + z * se)
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Path to human-labeled CSV")
    parser.add_argument(
        "--errors-out",
        default="human_eval_errors.csv",
        help="Where to save misclassified rows",
    )
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    train_script = os.path.join(here, "train_real_model.py")
    assets_dir = os.path.join(here, "..", "sms_fraud_detectore_app", "assets")
    tflite_path = os.path.join(assets_dir, "advanced_fraud_detector.tflite")
    config_path = os.path.join(assets_dir, "behavioral_model_config.json")

    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Missing model: {tflite_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")

    extract_features, label_message = load_feature_extractor(train_script)

    df = pd.read_csv(args.gold, dtype=str).fillna("")
    needed = {"sender", "body", "corrected_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df["corrected_label"] = df["corrected_label"].str.strip().str.lower()
    df = df[df["corrected_label"].isin(["legit", "spam", "fraud"])].copy()
    if len(df) == 0:
        raise ValueError("No rows with corrected_label in {legit, spam, fraud}")

    config = json.load(open(config_path, encoding="utf-8"))
    scaler_mean = np.array(config["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(config["scaler_scale"], dtype=np.float32)
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    input_idx = interp.get_input_details()[0]["index"]
    output_idx = interp.get_output_details()[0]["index"]

    label_to_id = {"legit": 0, "spam": 1, "fraud": 2}
    id_to_label = {0: "legit", 1: "spam", 2: "fraud"}

    y_true, y_pred = [], []
    probs = []
    auto_preds = []

    for _, row in df.iterrows():
        sender = str(row["sender"])
        body = str(row["body"])

        feats = np.array(extract_features(body, sender), dtype=np.float32)
        normed = (feats - scaler_mean) / scaler_scale

        inp = normed.reshape(1, -1).astype(np.float32)
        interp.set_tensor(input_idx, inp)
        interp.invoke()
        out = interp.get_tensor(output_idx)[0]

        pred_id = int(np.argmax(out))
        y_pred.append(pred_id)
        y_true.append(label_to_id[row["corrected_label"]])
        probs.append(float(out[pred_id]))
        auto_preds.append(label_message(sender, body))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float((y_true_arr == y_pred_arr).mean())
    lo, hi = accuracy_ci(acc, len(y_true_arr))

    print("=== Human-Labeled Evaluation ===")
    print(f"Samples: {len(y_true_arr)}")
    print(f"Accuracy: {acc:.4f}  (95% CI: {lo:.4f} - {hi:.4f})")
    print("\nClassification report:")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["LEGIT", "SPAM", "FRAUD"],
            digits=4,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true_arr, y_pred_arr))

    # Optional baseline: heuristic labeler agreement with same gold set.
    auto_ids = np.array([label_to_id[x] for x in auto_preds if x in label_to_id], dtype=np.int32)
    if len(auto_ids) == len(y_true_arr):
        auto_acc = float((auto_ids == y_true_arr).mean())
        print(f"\nHeuristic labeler accuracy on same gold set: {auto_acc:.4f}")

    out_df = df.copy()
    out_df["pred_label"] = [id_to_label[x] for x in y_pred]
    out_df["pred_confidence"] = probs
    out_df["auto_label"] = auto_preds
    err_df = out_df[out_df["pred_label"] != out_df["corrected_label"]].copy()
    err_df.to_csv(args.errors_out, index=False, encoding="utf-8")

    print(f"\nMisclassified rows saved: {args.errors_out} ({len(err_df)} rows)")


if __name__ == "__main__":
    main()
