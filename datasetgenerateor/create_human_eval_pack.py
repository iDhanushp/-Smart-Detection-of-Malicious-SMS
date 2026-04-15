"""Create a large human-review pack from real phone SMS exports.

Purpose:
- Expand manual evaluation from 117 samples to 500-1000+ samples.
- Build a reproducible, stratified review set with both balanced and prevalence slices.

Usage:
  python create_human_eval_pack.py --size 800 --seed 42

Output:
  human_eval_pack.csv
  human_eval_pack_summary.txt
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd


def load_labeler(train_script_path: str):
    """Load label_message() from train_real_model.py without running training."""
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
        "__name__": "__label_loader__",
    }
    exec(compile(src[:cut], train_script_path, "exec"), scope)
    if "label_message" not in scope:
        raise RuntimeError("label_message() not found after loading train_real_model.py")
    return scope["label_message"]


def sender_type(sender: str) -> str:
    s = str(sender or "").strip()
    if re.match(r"^\+\d{10,}$", s):
        return "phone"
    if "-" in s or len(s) <= 6:
        return "service"
    return "other"


def has_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.|wa\.me/|\.(com|in|org|io|co)\b", text.lower()))


def has_phone(text: str) -> bool:
    return bool(re.search(r"\b\d{10,}\b", text))


def risk_hits(text: str) -> int:
    t = text.lower()
    pats = [
        r"urgent|immediately|asap|expire|deadline",
        r"suspended|blocked|deactivated|penalty|fine|arrest|court",
        r"otp|pin|cvv|password|kyc|aadhaar|pan",
        r"lottery|prize|winner|cashback|bonus|jackpot",
        r"rummy|casino|poker|bet|satta|fantasy",
        r"income tax|rbi|sebi|government|police",
    ]
    return sum(1 for p in pats if re.search(p, t))


def sample_balanced(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    classes = ["legit", "spam", "fraud"]
    per_class = max(1, n // len(classes))

    out = []
    for c in classes:
        part = df[df["auto_label"] == c]
        if len(part) == 0:
            continue
        k = min(per_class, len(part))
        idx = rng.choice(part.index.to_numpy(), size=k, replace=False)
        out.append(part.loc[idx])

    if not out:
        return df.iloc[0:0]
    return pd.concat(out, ignore_index=True)


def sample_prevalence(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0]
    k = min(n, len(df))
    idx = rng.choice(df.index.to_numpy(), size=k, replace=False)
    return df.loc[idx].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=800, help="Total review pack size (default: 800)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="human_eval_pack.csv",
        help="Output CSV path (default: human_eval_pack.csv)",
    )
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "sms data set")
    train_script = os.path.join(here, "train_real_model.py")

    label_message = load_labeler(train_script)

    phone_csvs = sorted(
        [
            os.path.join(data_dir, x)
            for x in os.listdir(data_dir)
            if x.startswith("phone_sms_export_") and x.endswith(".csv")
        ]
    )
    if not phone_csvs:
        raise FileNotFoundError("No phone_sms_export_*.csv files found in sms data set/")

    frames = []
    for fp in phone_csvs:
        df = pd.read_csv(fp, dtype=str).fillna("")
        if "address" not in df.columns or "body" not in df.columns:
            continue
        sub = df[["address", "body"]].copy()
        sub["source_file"] = os.path.basename(fp)
        frames.append(sub)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"address": "sender"})
    raw["sender"] = raw["sender"].astype(str).str.strip()
    raw["body"] = raw["body"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    raw = raw[raw["body"].str.len() > 0]
    raw = raw.drop_duplicates(subset=["body"]).reset_index(drop=True)

    raw["sender_type"] = raw["sender"].map(sender_type)
    raw["auto_label"] = raw.apply(lambda r: label_message(r["sender"], r["body"]), axis=1)
    raw["has_url"] = raw["body"].map(has_url)
    raw["has_phone"] = raw["body"].map(has_phone)
    raw["risk_hits"] = raw["body"].map(risk_hits)

    raw["hard_case_score"] = (
        raw["risk_hits"].astype(int)
        + raw["has_url"].astype(int)
        + raw["has_phone"].astype(int)
        + (raw["sender_type"] == "phone").astype(int)
    )
    raw["priority"] = np.where(raw["hard_case_score"] >= 3, "high", "normal")

    rng = np.random.default_rng(args.seed)

    core_n = int(args.size * 0.7)
    tail_n = max(0, args.size - core_n)

    high_pool = raw[raw["priority"] == "high"]
    if len(high_pool) < core_n:
        core_pool = raw
    else:
        core_pool = high_pool

    balanced = sample_balanced(core_pool, core_n, rng)
    balanced["slice"] = "balanced_core"

    remaining = raw[~raw["body"].isin(set(balanced["body"]))]
    prevalence = sample_prevalence(remaining, tail_n, rng)
    prevalence["slice"] = "prevalence_tail"

    pack = pd.concat([balanced, prevalence], ignore_index=True)
    pack = pack.drop_duplicates(subset=["body"]).reset_index(drop=True)

    # Top-up to requested size when class-balanced sampling underfills
    # (common when one class, typically FRAUD, has low prevalence).
    if len(pack) < args.size:
        remainder = raw[~raw["body"].isin(set(pack["body"]))]
        need = min(args.size - len(pack), len(remainder))
        if need > 0:
            add_idx = rng.choice(remainder.index.to_numpy(), size=need, replace=False)
            add = remainder.loc[add_idx].copy().reset_index(drop=True)
            add["slice"] = "topup_tail"
            pack = pd.concat([pack, add], ignore_index=True)
            pack = pack.drop_duplicates(subset=["body"]).reset_index(drop=True)

    pack.insert(0, "sample_id", [f"HE-{i+1:04d}" for i in range(len(pack))])
    pack["corrected_label"] = ""
    pack["reviewer_id"] = ""
    pack["reviewed_at"] = ""
    pack["review_notes"] = ""

    keep_cols = [
        "sample_id",
        "sender",
        "body",
        "source_file",
        "sender_type",
        "auto_label",
        "priority",
        "hard_case_score",
        "slice",
        "corrected_label",
        "reviewer_id",
        "reviewed_at",
        "review_notes",
    ]
    pack = pack[keep_cols]
    pack.to_csv(args.out, index=False, encoding="utf-8")

    summary_path = os.path.splitext(args.out)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total rows: {len(pack)}\n")
        f.write(f"Source rows (deduped): {len(raw)}\n\n")
        f.write("Label distribution (auto_label):\n")
        for k, v in Counter(pack["auto_label"]).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nSender type distribution:\n")
        for k, v in Counter(pack["sender_type"]).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nPriority distribution:\n")
        for k, v in Counter(pack["priority"]).items():
            f.write(f"  {k}: {v}\n")

    print(f"Saved review pack: {args.out} ({len(pack)} rows)")
    print(f"Saved summary   : {summary_path}")
    print("Next: fill corrected_label with legit/spam/fraud and run evaluate_human_eval.py")


if __name__ == "__main__":
    main()
