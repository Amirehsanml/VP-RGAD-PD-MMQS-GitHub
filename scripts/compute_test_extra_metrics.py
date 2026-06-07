import argparse
import json
from pathlib import Path

import pandas as pd
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


def safe_download_nltk():
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)


def tokenize_for_meteor(text):
    return str(text).strip().split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file with actual and predicted columns")
    parser.add_argument("--ref_col", default="actual")
    parser.add_argument("--pred_col", default="predicted")
    parser.add_argument("--bert_model", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda or cpu. If omitted, bert-score decides.")
    parser.add_argument("--out_json", default="results/tables/test_extra_metrics.json")
    parser.add_argument("--out_csv", default="results/tables/test_extra_metrics.csv")
    args = parser.parse_args()

    safe_download_nltk()

    df = pd.read_csv(args.csv)

    if args.ref_col not in df.columns:
        raise ValueError(f"Missing reference column: {args.ref_col}. Found: {list(df.columns)}")
    if args.pred_col not in df.columns:
        raise ValueError(f"Missing prediction column: {args.pred_col}. Found: {list(df.columns)}")

    refs = df[args.ref_col].fillna("").astype(str).tolist()
    preds = df[args.pred_col].fillna("").astype(str).tolist()

    if len(refs) != len(preds):
        raise ValueError("Reference and prediction lengths do not match.")

    # BLEU: corpus-level BLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    # METEOR: sentence-level average
    meteor_scores = []
    for ref, pred in zip(refs, preds):
        meteor_scores.append(
            meteor_score([tokenize_for_meteor(ref)], tokenize_for_meteor(pred))
        )
    meteor = sum(meteor_scores) / len(meteor_scores) * 100

    # BERTScore
    bert_kwargs = {
        "lang": "en",
        "model_type": args.bert_model,
        "batch_size": args.batch_size,
        "verbose": True,
        "rescale_with_baseline": False,
    }
    if args.device:
        bert_kwargs["device"] = args.device

    P, R, F1 = bert_score(preds, refs, **bert_kwargs)

    metrics = {
        "num_samples": len(refs),
        "BLEU": float(bleu),
        "METEOR": float(meteor),
        "BERTScore_P": float(P.mean().item() * 100),
        "BERTScore_R": float(R.mean().item() * 100),
        "BERTScore_F1": float(F1.mean().item() * 100),
        "bert_model": args.bert_model,
        "input_csv": args.csv,
    }

    print(json.dumps(metrics, indent=2))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(args.out_csv, index=False)

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
