from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B/checkpoint-21077",
    torch_dtype="bfloat16",
    device_map="cpu"
)

# print all modules containing "distill"
for name, module in model.named_modules():
    if "distill" in name.lower() or "projection" in name.lower():
        print(name)