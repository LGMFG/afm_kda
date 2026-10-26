from datasets import load_dataset
ds = load_dataset("parquet",
    data_files=["/leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/fineweb_edu_raw/sample/100BT/000_00006.parquet"],
    split="train")
print(ds)                      # columns and row count
print(ds[0]["text"][:800])     # what one document looks like