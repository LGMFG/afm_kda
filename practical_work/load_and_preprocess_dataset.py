from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUT_PATH = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/tulu3_qwen06b_packed"

SEQ_LEN = 8192
MAX_PER_EXAMPLE = 4096   # safety cap (VERY important)
NUM_PROC = 32

# ---------------------------------------------------
# TOKENIZER
# ---------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EOS = tokenizer.eos_token_id

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

ds = load_dataset(
    "allenai/tulu-3-sft-mixture",
    split="train"
)

# optional but recommended for stability
ds = ds.shuffle(seed=42)

# ---------------------------------------------------
# STEP 1: TOKENIZE + CHAT TEMPLATE
# ---------------------------------------------------

def tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    # tokenize with HARD LIMIT
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_PER_EXAMPLE
    )["input_ids"]
    
    # ensure EOS at the end of each example (important for packing)
    ids.append(EOS)

    return {"input_ids": ids}

ds = ds.map(
    tokenize,
    remove_columns=ds.column_names,
    num_proc=NUM_PROC,
    desc="Tokenizing"
)

# ---------------------------------------------------
# STEP 2: STREAMING PACKING (NO BIG LIST IN RAM)
# ---------------------------------------------------

buffer = []
packed = []

for ids in tqdm(ds["input_ids"], desc="Packing"):
    buffer.extend(ids)

    while len(buffer) >= SEQ_LEN:
        chunk = buffer[:SEQ_LEN]
        packed.append(chunk)
        buffer = buffer[SEQ_LEN:]

# ---------------------------------------------------
# DROP REMAINDER (or keep if you want)
# ---------------------------------------------------

print(f"Dropped tail tokens: {len(buffer):,}")

# ---------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------

input_ids = np.array(packed, dtype=np.int32)
attention_mask = np.ones_like(input_ids, dtype=np.int8)

packed_ds = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

print(packed_ds)

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

packed_ds.save_to_disk(OUT_PATH)

print(f"Saved to {OUT_PATH}")