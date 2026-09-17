import os, torch
from transformers import AutoModelForCausalLM

P = f"{os.environ['WORK_LUKAS']}/students/afm_kda_only_init"
torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained(
        P, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

V, LA, LB = m.config.vocab_size, 301, 200
A  = torch.randint(0, V, (1, LA)).cuda()
A2 = torch.randint(0, V, (1, LA)).cuda()
B  = torch.randint(0, V, (1, LB)).cuda()

seg  = torch.cat([torch.ones(1, LA), 2 * torch.ones(1, LB)], 1).long().cuda()
ones = torch.ones(1, LA + LB).long().cuda()

@torch.no_grad()
def run(pre, mask):
    return m(input_ids=torch.cat([pre, B], 1),
             attention_mask=mask).logits[0, LA:, :].float()

for name, mask in (("segment-id mask [1..1,2..2]", seg),
                   ("all-ones mask   (control)  ", ones)):
    d = (run(A, mask) - run(A2, mask)).abs().max().item()
    print(f"{name}  B-after-A vs B-after-A': {d:.3e}")