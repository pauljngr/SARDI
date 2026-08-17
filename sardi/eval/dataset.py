# Multi-hop QA evaluation dataset.

import os
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset


class MultiHopQAEvalDataset(Dataset):
    """Concatenated parquet splits with columns question / answer / supporting_docs.

    `reasoning` is present in some splits and absent in others (CofCA and
    SynthWorlds have no gold reasoning traces); it is unused for Exact Match.
    """

    def __init__(self, data_paths: List[str], max_samples: Optional[int] = None):
        dfs = []
        for data_path in data_paths:
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Eval split not found: {data_path}\n"
                    "See the README section 'Assets' for the expected data layout."
                )
            dfs.append(pd.read_parquet(data_path))
        df = pd.concat(dfs, ignore_index=True)

        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=4711)

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        return {
            "id": row.get("id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "reasoning": row.get("reasoning"),
            "supporting_docs": row.get("supporting_docs"),
        }


def eval_collate(batch: List[Dict]) -> Dict:
    return {
        key: [x.get(key) for x in batch]
        for key in ("id", "question", "answer", "supporting_docs", "reasoning")
    }
