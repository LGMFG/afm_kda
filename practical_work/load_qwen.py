from transformers import AutoTokenizer, AutoModelForCausalLM

# !!! before running this:
# export HF_HOME=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
# export TRANSFORMERS_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache

model_id = "Qwen/Qwen3-1.7B-Base"
cache_dir = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache"

tok = AutoTokenizer.from_pretrained(
    model_id,
    #cache_dir=cache_dir,
    #local_files_only=True,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    #cache_dir=cache_dir,
    #local_files_only=True,
    trust_remote_code=True,
    #torch_dtype="auto",
    #device_map="auto"
)

print("loaded OK")
print(model.name_or_path)

from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")
print(len(tok.get_vocab()), max(tok.get_vocab().values()) + 1, cfg.vocab_size)
