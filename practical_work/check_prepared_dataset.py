from datasets import load_from_disk

path = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/prepared_datasets/tulu3/dataset-da644f1734d27818ab962ff7e81265210a9b3e45a77eb77b7013f15924dd7c4b"

ds = load_from_disk(path)

print(ds)
print(ds.column_names)
print(ds.features)
print(ds[0])