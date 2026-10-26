from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



base_model = "Qwen/Qwen3-0.6B"
#distilled_model = "./distillkit/output/tulu3_10k/checkpoint-274"
#distilled_model = "./distillkit/output/tulu3/checkpoint-25736"
#distilled_model = "./distillkit/output/tulu3_10k_lr1e-5_sl512/checkpoint-226"
#distilled_model = "./distillkit/output/tulu3_10k_lr1e-5_sl512_wd05_wr025_lrscos/checkpoint-226"
distilled_model = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_10k_4B_instr_06B/checkpoint-226"

tok = AutoTokenizer.from_pretrained(distilled_model, padding_side="left", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    distilled_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda",

)

prompts = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "2+2=",
    "Explain what a neural network is."
]

inputs = tok(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True
).to("cuda")

out = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False
)

for prompt, output_ids in zip(prompts, out):
    print("=" * 80)
    print("PROMPT:", prompt)
    print()
    print(tok.decode(output_ids, skip_special_tokens=True))
    print()