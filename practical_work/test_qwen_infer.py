import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"
#cache_dir = os.environ["HF_HOME"]
#cache_dir = "/leonardo_work//hf_cache"  # adjust if needed

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(
    model_id,
    #cache_dir=cache_dir,
    local_files_only=True,
    trust_remote_code=True,
    padding_side="left",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #cache_dir=cache_dir,
    local_files_only=True,
    trust_remote_code=True,

).to(device)

model.eval()


prompts = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "2+2=",
    "Explain what a neural network is."
]

inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)

for o in out:
    print(tok.decode(o, skip_special_tokens=True))