import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# CONFIG (CHANGE ONLY THIS SECTION)
# =========================================================

MODELS = {
    #"student_0.6b_base":
    #    "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/"
    #    "hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/"
    #    "snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",

    #"teacher_4b_instruct":
    #    "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/"
    #    "hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    #    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",

    "distilled_tulu":
        "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/"
        "distillkit/output/"
        "tulu3_4B_instr_06B_base_sl4096_bs1_gacc4_lr1e-4_"
        "mgn3_ce03_kl05_hsc02_t2/checkpoint-2000",

    #"distilled_dolci":
    #    "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/"
    #    "distillkit/output/"
    #    "dolci_4B_instr_06B_base_sl4096_bs1_gacc4_lr1e-4_"
    #    "mgn3_ce03_kl05_hsc02_t2/checkpoint-2000",
}

MODEL_PATH = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

HUMANEVAL_PATH = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/openai___openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"

NUM_TASKS = 1          # KEEP SMALL ON LOGIN NODE
MAX_NEW_TOKENS = 128

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading HumanEval...")

ds = load_dataset(
    "openai/openai_humaneval",
    cache_dir="/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache"
)

print(ds)

if isinstance(ds, dict) and "test" in ds:
    ds = ds["test"]

tasks = ds.select(range(NUM_TASKS))

print(f"Loaded {len(tasks)} tasks")

# =========================================================
# LOAD MODEL (ONE ONLY)
# =========================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="left",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=None,  # IMPORTANT: avoids login-node memory issues
    local_files_only=True,
)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Model loaded on {device}")

# =========================================================
# GENERATION FUNCTION
# =========================================================

@torch.no_grad()
def generate(prompt: str) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    return decoded[len(prompt):].strip()

# =========================================================
# RUN
# =========================================================

print("\n================ HUMAN EVAL RESULTS ================\n")

for i, ex in enumerate(tasks):

    task_id = ex["task_id"]
    prompt = ex["prompt"]

    print("\n" + "=" * 80)
    print(f"TASK {i+1}/{len(tasks)}: {task_id}")
    print("=" * 80)

    output = generate(prompt)

    print("\n--- PROMPT ---\n")
    print(prompt)

    print("\n--- OUTPUT ---\n")
    print(output)

print("\nDONE\n")