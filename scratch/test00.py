import datasets, numpy as np, os
from transformers import AutoTokenizer

W = os.environ["WORK_LUKAS"]
tok = AutoTokenizer.from_pretrained("arcee-ai/AFM-4.5B-Base")
ds = datasets.load_from_disk(f"{W}/datasets/fineweb_edu_3B_train").select(range(2000))
print("columns:", ds.column_names)

texts = [t for t in list(ds["text"]) if isinstance(t, str) and t]
print(f"{len(texts)} usable of {len(ds)}")

L = np.array([len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]])
print(f"mean {L.mean():.0f}   median {np.median(L):.0f}   p95 {np.percentile(L,95):.0f}   max {L.max()}")
print(f"docs over 2048: {(L > 2048).mean():.1%}")
print(f"tokens lost if truncated at 2048: {1 - np.minimum(L, 2048).sum() / L.sum():.1%}")