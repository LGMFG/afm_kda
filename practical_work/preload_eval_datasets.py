from datasets import load_dataset

CACHE_DIR = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache"

datasets_to_load = [
    # ARC
    #("allenai/ai2_arc", "ARC-Easy"),
    #("allenai/ai2_arc", "ARC-Challenge"),

    # HellaSwag
    #("Rowan/hellaswag", None),

    # WinoGrande
    #("allenai/winogrande", "winogrande_xl"),

    # GSM8K
    #("openai/gsm8k", "main"),

    # MMLU (large but correct canonical config)
    #("cais/mmlu", None),

    # humaneval
    #("openai/openai_humaneval", None),

    #mbpp
    #("google-research-datasets/mbpp", "full"),

    #mgsm
    #("juletxara/mgsm", "de"),

    #ifeval
    ("google/IFEval", None)
]

for item in datasets_to_load:
    dataset_name, config = item

    print(f"\nLoading {dataset_name} / {config}")

    if config is None:
        ds = load_dataset(
            dataset_name,
            cache_dir=CACHE_DIR
        )
    else:
        ds = load_dataset(
            dataset_name,
            config,
            cache_dir=CACHE_DIR
        )

    print(ds)