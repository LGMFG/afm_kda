import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-4B-Instruct-2507"
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
    torch_dtype="auto",
    device_map="auto"

).to(device)

model.eval()


prompt = "What is the capital of France?"

messages = [{"role": "user", "content": prompt}]

text = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tok(text, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=100
)

print(tok.decode(out[0], skip_special_tokens=True))