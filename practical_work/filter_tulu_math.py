from datasets import load_from_disk, load_dataset

DATASET_PATH = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/tulu3"

ds = load_from_disk(DATASET_PATH)

#ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", cache_dir="/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache")

print(sorted(set(ds["source"])))                      # eyeball the list once

keys = ("math", "gsm", "numina", "grade", "algebra", "geometry", "calculus", "trigonometry", "probability", "statistics", "number_theory", "combinatorics", "linear_algebra")
math_sources = {s for s in set(ds["source"]) if any(k in s.lower() for k in keys)}
print(math_sources)                                   # sanity-check what you caught
# print number of math_sources and total number of sources
print(f"Number of math sources: {len(math_sources)}")
print(f"Total number of sources: {len(set(ds['source']))}")

ds_math = ds.filter(lambda ex: ex["source"] in math_sources, num_proc=32)
ds_math.save_to_disk("/leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/tulu3_math")