"""Build data/questions.jsonl from MMLU (test) and ARC-Challenge (test) parquet files.

    curl -L -o data/mmlu_test.parquet https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet
    curl -L -o data/arc_test.parquet  https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet
    python src/fetch_data.py
"""
import json
import random
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"


def main(seed=0, n_mmlu=400, n_arc=200):
    rnd = random.Random(seed)
    items = []
    m = pd.read_parquet(D / "mmlu_test.parquet")
    for i in rnd.sample(range(len(m)), n_mmlu):
        x = m.iloc[i]
        items.append({"id": f"mmlu-{i}", "source": "mmlu", "subject": x["subject"], "question": x["question"],
                      "choices": list(x["choices"]), "answer": int(x["answer"])})
    a = pd.read_parquet(D / "arc_test.parquet")
    idx = [i for i in range(len(a)) if len(a.iloc[i]["choices"]["label"]) == 4]
    for i in rnd.sample(idx, n_arc):
        x = a.iloc[i]
        labels = list(x["choices"]["label"])
        items.append({"id": f"arc-{x['id']}", "source": "arc", "subject": "science", "question": x["question"],
                      "choices": list(x["choices"]["text"]), "answer": labels.index(x["answerKey"])})
    (D / "questions.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    print(len(items), "questions")


if __name__ == "__main__":
    main()
