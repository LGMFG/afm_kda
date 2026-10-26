import json, glob, numpy as np
from transformers import AutoTokenizer

# IFEval generations are greedy; we re-tokenize the output text to count tokens.
tok = AutoTokenizer.from_pretrained(
    "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
)  # any Qwen3 snapshot works — shared tokenizer

def gen_text(d):
    g = d.get("filtered_resps") or d.get("resps")
    # unwrap nested lists like [["text"]] or ["text"]
    while isinstance(g, list) and g:
        g = g[0]
    return g if isinstance(g, str) else str(g)

CAP = 1280
for name, path in [("base","/leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_results/qwen0.6b_base_ifeval_chat_1280/__leonardo_work__EUHPC_D34_049__WORK_LUKAS__hf_cache__hub__models--Qwen--Qwen3-0.6B-Base__snapshots__da87bfb608c14b7cf20ba1ce41287e8de496c0cd/samples_ifeval_2026-06-21T01-26-57.297276.jsonl"),
                    ("base_nochat","/leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_results/qwen0.6b_base_ifeval_nochat/__leonardo_work__EUHPC_D34_049__WORK_LUKAS__hf_cache__hub__models--Qwen--Qwen3-0.6B-Base__snapshots__da87bfb608c14b7cf20ba1ce41287e8de496c0cd/samples_ifeval_2026-06-20T23-43-21.745911.jsonl"),
                   ("dolci","/leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_results/dolci_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu_ifeval_chat_1280/__leonardo_work__EUHPC_D34_049__WORK_LUKAS__distillkit__output__dolci_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu__checkpoint-19348/samples_ifeval_2026-06-21T14-15-53.679812.jsonl"),
                   ("teacher","/leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_results/qwen4b_instr_ifeval_chat/__leonardo_work__EUHPC_D34_049__WORK_LUKAS__hf_cache__hub__models--Qwen--Qwen3-4B-Instruct-2507__snapshots__cdbee75f17c01a7cc42f958dc650907174af0554/samples_ifeval_2026-06-20T17-39-36.884482.jsonl")]:
    toks = []
    for line in open(path):
        d = json.loads(line)
        toks.append(len(tok.encode(gen_text(d), add_special_tokens=False)))
    toks = np.array(toks)
    print(f"{name:8s} median_tok {int(np.median(toks)):5d}  "
          f"mean_tok {toks.mean():6.1f}  frac_at_cap {np.mean(toks >= CAP*0.98):.2f}") # frac 
    
    print("max:", toks.max(), "frac==2048:", np.mean(toks==2048), "frac<500:", np.mean(toks<500))