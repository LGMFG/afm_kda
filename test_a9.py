import os, torch, transformers
from transformers import AutoModelForCausalLM
from distillkit.monkey_patch_packing import get_unpad_data
transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

T = "arcee-ai/AFM-4.5B-Base"
m = AutoModelForCausalLM.from_pretrained(
        T, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda().eval()

torch.manual_seed(0)
V, LA, LB = m.config.vocab_size, 301, 200
A  = torch.randint(0, V, (1, LA)).cuda()
A2 = torch.randint(0, V, (1, LA)).cuda()          # a DIFFERENT document A'
B  = torch.randint(0, V, (1, LB)).cuda()

seg  = torch.cat([torch.ones(1, LA), 2 * torch.ones(1, LB)], 1).long().cuda()
ones = torch.ones(1, LA + LB).long().cuda()

def tail(first, mask):
    ids = torch.cat([first, B], 1)
    with torch.no_grad():
        return m(input_ids=ids, attention_mask=mask).logits[0, -LB:].float()

for name, mask in (("segment-id mask", seg), ("all-ones control", ones)):
    d = (tail(A, mask) - tail(A2, mask)).abs().max().item()
    print(f"{name:18s} B-after-A vs B-after-A': {d:.3e}")