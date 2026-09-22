#!/usr/bin/env python3
"""Build a regression corpus from an sttd dump dir.

`sttd --dump-dir D` writes `utt_NNNNNN.wav` plus a sibling `.txt` per
utterance: the transcript for emits, `DROPPED(<reason>)...` for drops.
This tool snapshots that state into an `eval.py` corpus (`NNN.wav` +
`NNN.txt` pairs):

- emit `.txt` (not starting with `DROPPED`) -> reference = transcript.
  Future runs must reproduce it (characterization lock against
  regressions and accidental retunes).
- drop `.txt` (`DROPPED...`) -> reference = empty. These are the FAR
  set: the dumped audio once produced no transcript, and a retune that
  starts emitting for it shows up as a false accept.

WAVs are copied byte-identical (dumps are already 16 kHz mono, the same
shape `--file` accepts). Pairs sort by dump tag so numbering is stable;
output names are zero-padded (`001`, `002`, ...) regardless of the
original `utt_` numbers.

Example::

    tools/dump2corpus.py --dumps dumps/2026-09-22 --corpus corpus/regression/

Stdlib only. Self-test: `tools/dump2corpus.py --self-test`.
"""

import argparse
import shutil
import sys
from pathlib import Path


def is_drop_txt(text):
    return text.lstrip().startswith("DROPPED")


def build_corpus(dumps, corpus):
    """Copy dump pairs into corpus layout. Returns (n_emit, n_drop, skipped)."""
    dumps = Path(dumps)
    corpus = Path(corpus)
    wavs = sorted(dumps.glob("utt_*.wav"))
    if not wavs:
        raise SystemExit(f"no utt_*.wav in {dumps}")
    corpus.mkdir(parents=True, exist_ok=True)
    n_emit = n_drop = 0
    skipped = []
    for i, wav in enumerate(wavs, 1):
        txt_path = wav.with_suffix(".txt")
        if not txt_path.exists():
            skipped.append(f"{wav.name}: no {txt_path.name}")
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        name = f"{i:03d}"
        shutil.copy(wav, corpus / f"{name}.wav")
        if is_drop_txt(text):
            (corpus / f"{name}.txt").write_text("", encoding="utf-8")
            n_drop += 1
        else:
            (corpus / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8")
            n_emit += 1
    return n_emit, n_drop, skipped


def self_test():
    import tempfile
    assert is_drop_txt("DROPPED(low-conf avg_p=0.1 < 0.6, weak): hej")
    assert is_drop_txt("  DROPPED(empty)")
    assert not is_drop_txt("Dzień dobry")
    assert not is_drop_txt("")
    with tempfile.TemporaryDirectory() as tmp:
        dumps = Path(tmp) / "dumps"
        dumps.mkdir()
        (dumps / "utt_000001.wav").write_bytes(b"RIFF" + b"\0" * 44)
        (dumps / "utt_000001.txt").write_text("Dzień dobry", encoding="utf-8")
        (dumps / "utt_000002.wav").write_bytes(b"RIFF" + b"\0" * 44)
        (dumps / "utt_000002.txt").write_text("DROPPED(hallucination): napisy", encoding="utf-8")
        (dumps / "utt_000003.wav").write_bytes(b"RIFF" + b"\0" * 44)
        # No txt for 003 -> skipped.
        corpus = Path(tmp) / "corpus"
        n_emit, n_drop, skipped = build_corpus(dumps, corpus)
        assert (n_emit, n_drop) == (1, 1), (n_emit, n_drop)
        assert len(skipped) == 1, skipped
        assert (corpus / "001.txt").read_text(encoding="utf-8").strip() == "Dzień dobry"
        assert (corpus / "002.txt").read_text(encoding="utf-8") == ""
        assert (corpus / "001.wav").exists() and (corpus / "002.wav").exists()
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=False, help="dir with utt_*.wav + utt_*.txt")
    ap.add_argument("--corpus", required=False, help="output dir for NNN.wav + NNN.txt")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.dumps or not args.corpus:
        ap.error("--dumps and --corpus are required (or use --self-test)")
    n_emit, n_drop, skipped = build_corpus(args.dumps, args.corpus)
    for s in skipped:
        print(f"  skip {s}", file=sys.stderr)
    print(f"wrote {args.corpus}: {n_emit} emit refs, {n_drop} empty (FAR) refs, "
          f"{len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
