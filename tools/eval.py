#!/usr/bin/env python3
"""Batch eval for sttd: WER over a WAV/TXT corpus + threshold sweep.

Corpus layout (flat dir)::

    corpus/
      001.wav + 001.txt   # reference transcript, plain text
      002.wav + 002.txt
      ...

Each `.wav` is decoded offline via `sttd --file <wav> --stdout` (copies,
pastes and sounds forced off for determinism); stdout lines (one per
emitted utterance) are joined with a space as the hypothesis. Reference
and hypothesis go through the same normalization as the transcript gate
(`gate::normalize_transcript`: lowercase, fold Polish diacritics,
punctuation -> space, collapse whitespace), then standard word-level
Levenshtein gives S/D/I and WER = (S+D+I) / N.

Sweep example (cartesian product, `;` between vars, `,` between values)::

    tools/eval.py --corpus corpus/ --output eval-out/ \\
      --sweep "STTD_MIN_CONF=0.4,0.6;STTD_MIN_DENSITY=0.2,0.3" \\
      --extra-args --model small --language pl

Output in `--output`: `results.jsonl` (one row per file per combo) and
`summary.csv` (one row per combo: count, avg/median/p90 WER, empty rates).
A summary table is also printed to stdout.

Sequential by design: every `--file` run loads the whisper model (GPU
VRAM), so parallel jobs would OOM or thrash. Model load dominates small
corpora — that is expected, not a harness bug.

Stdlib only. Self-test: `tools/eval.py --self-test`.
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Must mirror gate::fold_polish_diacritic (lowercased input).
DIACRITIC_MAP = {
    "ą": "a",
    "ć": "c",
    "ę": "e",
    "ł": "l",
    "ń": "n",
    "ó": "o",
    "ś": "s",
    "ź": "z",
    "ż": "z",
}


def normalize(text, fold_diacritics=True):
    """Mirror gate::normalize_transcript."""
    text = text.lower()
    if fold_diacritics:
        text = "".join(DIACRITIC_MAP.get(c, c) for c in text)
    text = "".join(c if c.isalnum() else " " for c in text)
    return " ".join(text.split())


def wer_details(ref_words, hyp_words):
    """Word-level Levenshtein. Returns (s, d, i)."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return (0, 0, m)
    if m == 0:
        return (0, n, 0)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    # Backtrack for S/D/I split (sub-optimal ties broken I > D > S order).
    i, j = n, m
    # Rebuild full matrix for backtrack (corpora are short; clarity first).
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        dp[a][0] = a
    for b in range(1, m + 1):
        dp[0][b] = b
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            cost = 0 if ref_words[a - 1] == hyp_words[b - 1] else 1
            dp[a][b] = min(dp[a - 1][b] + 1, dp[a][b - 1] + 1, dp[a - 1][b - 1] + cost)
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            d += 1
            i -= 1
    return (s, d, ins)


def score_pair(ref_text, hyp_text):
    ref_norm = normalize(ref_text)
    hyp_norm = normalize(hyp_text)
    ref_w = ref_norm.split() if ref_norm else []
    hyp_w = hyp_norm.split() if hyp_norm else []
    if not ref_w:
        w = 0.0 if not hyp_w else 1.0
        return {"wer": w, "s": 0, "d": 0, "i": len(hyp_w), "n": 0,
                "ref_norm": ref_norm, "hyp_norm": hyp_norm}
    s, d, ins = wer_details(ref_w, hyp_w)
    return {"wer": (s + d + ins) / len(ref_w), "s": s, "d": d, "i": ins,
            "n": len(ref_w), "ref_norm": ref_norm, "hyp_norm": hyp_norm}


def parse_sweep(spec):
    """'A=1,2;B=3' -> [{'A':'1','B':'3'}, ...]. Empty spec -> [{}]."""
    if not spec or not spec.strip():
        return [{}]
    groups = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        var, _, vals = chunk.partition("=")
        var = var.strip()
        vals = [v.strip() for v in vals.split(",") if v.strip() != ""]
        if not var or not vals:
            raise ValueError(f"bad sweep chunk: {chunk!r} (want VAR=v1,v2)")
        groups.append([(var, v) for v in vals])
    if not groups:
        return [{}]
    return [dict(combo) for combo in itertools.product(*groups)]


def run_file(binary, wav, extra_args, env_overrides, timeout):
    env = dict(os.environ)
    env.update({
        "STTD_COPY": "0",
        "STTD_PASTE": "0",
        "STTD_SOUND": "0",
    })
    env.update(env_overrides)
    cmd = [str(binary), "--file", str(wav), "--stdout"] + extra_args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "hyp": "", "utterances": 0,
                "error": f"timeout after {timeout}s"}
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        return {"ok": False, "hyp": "", "utterances": 0,
                "error": f"exit {p.returncode}: " + " | ".join(tail)}
    lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    return {"ok": True, "hyp": " ".join(lines),
            "utterances": len(lines), "error": ""}


def percentile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[k]


def self_test():
    assert normalize("  Napisy... Amara.org! ") == "napisy amara org"
    assert normalize("Dziękuje za UWAGĘ.") == "dziekuje za uwage"
    assert normalize("Zrób-commit_repo") == "zrob commit repo"
    r = score_pair("kot pies dom", "kot pies dom")
    assert r["wer"] == 0.0, r
    r = score_pair("kot pies dom", "kot dom")
    assert r["wer"] == 1 / 3, r
    r = score_pair("kot", "pies")
    assert r["wer"] == 1.0, r
    r = score_pair("", "")
    assert r["wer"] == 0.0, r
    r = score_pair("", "hej")
    assert r["wer"] == 1.0, r
    r = score_pair("Dziękuję za uwagę", "dziekuje za uwage")
    assert r["wer"] == 0.0, r
    assert parse_sweep("") == [{}]
    assert parse_sweep("A=1,2;B=x") == [
        {"A": "1", "B": "x"}, {"A": "2", "B": "x"}]
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=False, help="dir with *.wav + *.txt pairs")
    ap.add_argument("--binary", default="target/debug/sttd")
    ap.add_argument("--output", default="eval-out")
    ap.add_argument("--sweep", default="",
                    help="'VAR=v1,v2;VAR2=v3' grid over env (cartesian)")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="appended to sttd cmdline; use '-- ...' separator")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    if not args.corpus:
        ap.error("--corpus is required (or use --self-test)")
    corpus = Path(args.corpus)
    binary = Path(args.binary)
    outdir = Path(args.output)
    if not binary.exists():
        ap.error(f"binary not found: {binary} (build first: cargo build)")
    wavs = sorted(corpus.glob("*.wav"))
    if not wavs:
        ap.error(f"no *.wav in {corpus}")
    extra = [a for a in args.extra_args if a != "--"]

    combos = parse_sweep(args.sweep)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    summary_path = outdir / "summary.csv"

    summaries = []
    with open(results_path, "w", encoding="utf-8") as jf:
        for ci, combo in enumerate(combos):
            label = ",".join(f"{k}={v}" for k, v in sorted(combo.items())) or "base"
            print(f"[{ci + 1}/{len(combos)}] {label} ({len(wavs)} files)")
            wers, n_empty_ref, n_empty_hyp = [], 0, 0
            for wav in wavs:
                ref_path = wav.with_suffix(".txt")
                if not ref_path.exists():
                    print(f"  skip {wav.name}: no {ref_path.name}", file=sys.stderr)
                    continue
                ref = ref_path.read_text(encoding="utf-8", errors="replace")
                run = run_file(binary, wav, extra, combo, args.timeout)
                s = score_pair(ref, run["hyp"])
                if not ref.strip():
                    n_empty_ref += 1
                if not run["hyp"].strip():
                    n_empty_hyp += 1
                wers.append(s["wer"])
                jf.write(json.dumps({
                    "combo": combo, "label": label, "file": wav.name,
                    "ok": run["ok"], "error": run["error"],
                    "utterances": run["utterances"],
                    "wer": round(s["wer"], 4), "s": s["s"], "d": s["d"],
                    "i": s["i"], "n": s["n"],
                    "ref": ref.strip(), "hyp": run["hyp"].strip(),
                }, ensure_ascii=False) + "\n")
            n = len(wers)
            avg = sum(wers) / n if n else 0.0
            med = percentile(wers, 0.5)
            p90 = percentile(wers, 0.9)
            summaries.append({"label": label, "combo": combo, "count": n,
                              "avg_wer": round(avg, 4), "median_wer": round(med, 4),
                              "p90_wer": round(p90, 4),
                              "empty_ref": n_empty_ref, "empty_hyp": n_empty_hyp})
            print(f"  count={n} avg={avg:.3f} median={med:.3f} p90={p90:.3f} "
                  f"empty_hyp={n_empty_hyp}")

    with open(summary_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=["label", "count", "avg_wer",
                                           "median_wer", "p90_wer",
                                           "empty_ref", "empty_hyp", "combo"])
        w.writeheader()
        for s in summaries:
            w.writerow({**s, "combo": json.dumps(s["combo"], ensure_ascii=False)})

    print(f"\nwrote {results_path} + {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
