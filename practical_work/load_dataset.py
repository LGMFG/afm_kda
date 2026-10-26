from datasets import load_dataset

#ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")

#ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")

ds = load_dataset("meta-math/MetaMathQA", split="train")   # columns: query, response
def to_messages(ex):
    return {"messages": [
        {"role": "user", "content": ex["query"]},
        {"role": "assistant", "content": ex["response"]},
    ]}
ds = ds.map(to_messages, remove_columns=ds.column_names, num_proc=32)
ds.save_to_disk("/leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/metamathqa")

# take subset
#ds_small = ds.shuffle(seed=42).select(range(10000))

#ds_small.save_to_disk("./datasets/tulu3_10k")