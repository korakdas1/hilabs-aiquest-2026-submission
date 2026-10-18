"""Captioned, synthetic live demo suitable for a screen recording.

The timed mode runs the real local CLI on examples/hard_cases.csv. It does not
open private training records or send text to an API. Use a 1280x720 display.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from tkinter import BOTH, END, Button, Frame, Label, Text, Tk

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
BG = "#102A3A"
PANEL = "#18394B"
FG = "#EDF5F6"
TEAL = "#57D7CA"
MUTED = "#AFCDD2"


def run_cli(*args: str) -> str:
    result = subprocess.run([str(PYTHON), "-m", "termmap.cli", *args],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


class Demo:
    def __init__(self, root: Tk, auto: bool):
        self.root = root
        self.auto = auto
        root.title("TermMap | Local clinical term mapping demo")
        root.geometry("1280x720+0+0")
        root.configure(bg=BG)
        root.resizable(False, False)

        head = Frame(root, bg=BG)
        head.pack(fill="x", padx=44, pady=(27, 13))
        Label(head, text="TermMap", font=("DejaVu Sans", 28, "bold"),
              fg=FG, bg=BG).pack(side="left")
        Label(head, text="ICD-10-CM  +  SNOMED CT", font=("DejaVu Sans", 13),
              fg=TEAL, bg=BG).pack(side="right", pady=12)
        self.section = Label(root, text="", font=("DejaVu Sans", 15, "bold"),
                             fg=TEAL, bg=BG, anchor="w")
        self.section.pack(fill="x", padx=44, pady=(0, 10))
        self.screen = Text(root, bg=PANEL, fg=FG, insertbackground=FG,
                           font=("DejaVu Sans Mono", 15), wrap="word",
                           relief="flat", padx=23, pady=20, height=14,
                           highlightthickness=1, highlightbackground="#326073")
        self.screen.pack(fill=BOTH, expand=True, padx=44, pady=(0, 14))
        self.screen.configure(state="disabled")
        self.caption = Label(root, text="", font=("DejaVu Sans", 13), fg=FG,
                             bg=BG, anchor="w", justify="left", wraplength=1180)
        self.caption.pack(fill="x", padx=44, pady=(0, 16))
        if auto:
            self.schedule()
        else:
            self.intro()
            Button(root, text="Next", command=self.next_scene).pack()
        self.step = 0

    def show(self, title: str, body: str, caption: str):
        self.section.config(text=title)
        self.screen.configure(state="normal")
        self.screen.delete("1.0", END)
        self.screen.insert("1.0", body)
        self.screen.configure(state="disabled")
        self.caption.config(text=caption)

    def intro(self):
        self.show("01 / Architecture",
                  "INPUT: clinical entity + optional chart chunk\n"
                  "   |\n"
                  "   v\n"
                  "NORMALIZE: Unicode, punctuation, stable shorthand\n"
                  "   |\n"
                  "   v\n"
                  "RETRIEVE: SQLite FTS5 + training-example TF-IDF\n"
                  "   |\n"
                  "   v\n"
                  "RERANK: separate local LightGBM models\n"
                  "   |\n"
                  "   v\n"
                  "VALIDATE: both IDs in supplied legal-code workbooks\n"
                  "   |\n"
                  "   v\n"
                  "OUTPUT: chunk_id, clinical_entity, icd10cm_code, snomed_code",
                  "The submitted path is local and deterministic. No API key or hosted model is needed.")

    def input_scene(self):
        content = (ROOT / "examples/hard_cases.csv").read_text(encoding="utf-8")
        self.show("02 / Synthetic live input",
                  "$ cat examples/hard_cases.csv\n\n" + content,
                  "The four cases probe abbreviation collision, OCR-like noise, laterality, and specificity. No patient data appears in this demo.")

    def batch_scene(self):
        command = (".venv/bin/python -m termmap.cli predict "
                   "--input examples/hard_cases.csv --output predictions_demo.csv")
        result = run_cli("predict", "--input", "examples/hard_cases.csv",
                         "--output", "predictions_demo.csv")
        content = (ROOT / "predictions_demo.csv").read_text(encoding="utf-8")
        self.show("03 / Live batch prediction",
                  f"$ {command}\n{result}\n\n$ cat predictions_demo.csv\n{content}",
                  "One command maps every row to both code systems. The output preserves IDs and follows the exact required schema.")

    def failure_scene(self):
        command = (".venv/bin/python -m termmap.cli demo --term 'MS' "
                   "--context 'Neurology follow-up for multiple sclerosis with gait difficulty.'")
        result = json.loads(run_cli("demo", "--term", "MS", "--context",
                                    "Neurology follow-up for multiple sclerosis with gait difficulty."))
        lines = [f"$ {command}", "", "Local decision:",
                 f"  ICD-10-CM   {result['prediction']['ICD10CM']}",
                 f"  SNOMED CT   {result['prediction']['SNOMEDCT_US']}", "",
                 "Top ICD candidates:"]
        for rank, item in enumerate(result["candidates"]["ICD10CM"][:3], 1):
            lines.append(f"  {rank}. {item['code']:<9} {item['description'][:55]}")
        lines.extend(["", "Known failure:",
                      "  The chart context says multiple sclerosis,",
                      "  but the local ICD code is altered mental status.",
                      "  Candidate scores rank options; they are not probabilities."])
        self.show("04 / Inspect a hard failure", "\n".join(lines),
                  "This is an honest error example. The short abbreviation overpowered its neurological context.")

    def evidence_scene(self):
        with (ROOT / "predictions_demo.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        lines = ["Other synthetic cases:", ""]
        for row in rows[1:]:
            lines.extend([f"  {row['clinical_entity']}",
                          f"    ICD {row['icd10cm_code']}  |  SNOMED {row['snomed_code']}", ""])
        lines.extend(["Grouped holdout exact accuracy:",
                      "  Local seed 42:  ICD 48.0%   SNOMED 48.4%",
                      "  Local seed 43:  ICD 52.8%   SNOMED 49.3%", "",
                      "Optional DeepSeek 1017 experiment, seed 42:",
                      "  ICD 65.2%   SNOMED 61.1%", "",
                      "No measured path reached the 95% challenge target."])
        self.show("05 / Evidence and limits", "\n".join(lines),
                  "The hosted result is reported for context only. It is excluded from the submitted deterministic command.")

    def close_scene(self):
        self.show("06 / Reproduce the submission output",
                  "Place the three supplied challenge files in this repository.\n\n"
                  "$ python3 -m venv .venv\n"
                  "$ .venv/bin/python -m pip install -r requirements.txt\n\n"
                  "$ .venv/bin/python -m termmap.cli predict \\\n"
                  "    --input evaluation.csv --output predictions.csv\n\n"
                  "Required output columns:\n"
                  "chunk_id,clinical_entity,icd10cm_code,snomed_code\n\n"
                  "The local index and model are built automatically on first use.",
                  "See README.md for setup, MODEL_CARD.md for validation, and the PDF report for error analysis.")

    def schedule(self):
        scenes = [(0, self.intro), (25, self.input_scene), (50, self.batch_scene),
                  (78, self.failure_scene), (108, self.evidence_scene),
                  (132, self.close_scene)]
        for seconds, fn in scenes:
            self.root.after(seconds * 1000, fn)
        self.root.after(151_000, self.root.destroy)

    def next_scene(self):
        scenes = [self.input_scene, self.batch_scene, self.failure_scene,
                  self.evidence_scene, self.close_scene]
        if self.step < len(scenes):
            scenes[self.step]()
            self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Run timed 2.5-minute captioned sequence")
    args = parser.parse_args()
    root = Tk()
    Demo(root, auto=args.auto)
    root.mainloop()


if __name__ == "__main__":
    main()
