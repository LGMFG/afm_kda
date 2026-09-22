# The training sbatch, explained line by line

*Reference for `claude/startup_plan_and_pitfalls.md` Part 3. Written 2026-09-11.*

---

## 1. Four programs, stacked

The confusing part is that four different things each launch something.

| Layer | What it is | What it launches |
|---|---|---|
| **SLURM** (`sbatch`) | the cluster scheduler | runs your script **once**, on the first allocated node |
| **`srun`** | SLURM's task launcher | one task per node (`--ntasks-per-node=1`) |
| **`accelerate launch`** | HuggingFace's process launcher | one Python process **per GPU**, on its own node |
| **DeepSpeed** | a library inside those processes | nothing — it does ZeRO memory sharding |

At 4 nodes: SLURM allocates 4 machines → the script runs on machine 0 → `srun` starts 4 tasks → each task's `accelerate launch` spawns 4 processes → **16 Python processes, one per GPU.**

⚠️ **Everything before the `srun` line runs on node 0 only.** That is why the GPU monitor block only ever produces one log file regardless of node count. To sample every node, move the monitor inside the `srun bash -c` body.

---

## 2. The `#SBATCH` lines

```bash
#SBATCH -A EUHPC_D34_049      # which budget to bill
#SBATCH -p boost_usr_prod     # which pool of machines (the GPU partition)
#SBATCH --qos=boost_qos_lprod # priority class: sets walltime + size limits
#SBATCH --nodes=4             # how many machines
#SBATCH --ntasks-per-node=1   # how many srun tasks per machine
#SBATCH --gres=gpu:4          # how many GPUs per machine
#SBATCH --cpus-per-task=32    # CPU cores per task
#SBATCH --time=24:00:00       # walltime; the job is KILLED at this limit
#SBATCH -o .../logs/%x-%j.out # stdout  (%x = job name, %j = job id)
#SBATCH -e .../logs/%x-%j.err # stderr
```

**`-A`** — without it, SLURM bills whichever account it picks. Works today because you have one; silently wrong the moment that changes.

**`--qos`** — not just priority. It also caps walltime and job size. `boost_qos_dbg` is 30 min / priority 80 / max 2 concurrent (fast queue, for debugging). `boost_qos_lprod` is 4 days / priority 40 / up to 8 nodes (real runs).

**`--ntasks-per-node=1` is not a typo.** You have 4 GPUs per node, but `accelerate` does the per-GPU spawning. Set this to 4 and SLURM starts four tasks that each start four processes — sixteen processes fighting over four GPUs. It hangs rather than erroring.

**`--cpus-per-task=32`** — the whole node. Leonardo bills the *largest* fraction of the node you occupy, so 4/4 GPUs already means you pay for the whole machine and the cores are free. The trap is the reverse case: 1 GPU + 32 cores bills at the full-node rate because the CPU fraction dominates. If you ever request 1 GPU, request ≤ 8 cores.

**`--time`** — the job is killed at this limit, not paused. Always leave headroom, and make sure checkpointing works (step 7c).

---

## 3. The rendezvous block

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))
```

**What a rendezvous is.** Sixteen processes on four machines have to find each other and agree on who is rank 0 through 15. One machine and port act as a meeting point: every process connects there at startup, announces itself, and is told its rank plus where everyone else is. After that they talk directly over InfiniBand and the meeting point stops mattering.

`$SLURM_JOB_NODELIST` is a compressed string like `lrdn[0703,3079,3167,3211]`. `scontrol show hostnames` expands it one per line; `head -n1` takes the first. **Every node computes the same answer**, because they all see the same nodelist — that is what makes them agree.

⚠️ **Do not name a variable `WORLD_SIZE`.** torch.distributed reads that name from the environment itself, so your value competes with the launcher's. Call it `TOTAL_PROCS`.

### The launch line

```bash
srun bash -c '
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}
  export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}_${SLURM_PROCID}
  accelerate launch \
    --use_deepspeed \
    --deepspeed_multinode_launcher standard \
    --deepspeed_config_file $DS_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $TOTAL_PROCS \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $(which distillkit) -v $RUN_CONFIG
'
```

| Flag | Meaning |
|---|---|
| `--use_deepspeed` | use DeepSpeed rather than plain DDP or FSDP |
| `--deepspeed_multinode_launcher standard` | **mandatory for multi-node.** Use the torch.distributed rendezvous below. Without it accelerate hands off to DeepSpeed's own launcher, which without a hostfile builds a world from the local node only — every node runs an independent job. |
| `--deepspeed_config_file` | the ZeRO JSON |
| `--num_machines` | how many nodes are in the group |
| `--num_processes` | how many processes total (16 at 4 nodes) |
| `--machine_rank` | **which node this one is** — the only value that differs per node |
| `--main_process_ip/port` | the rendezvous meeting point |
| `-v` | INFO logging; `main.py` defaults to WARNING and you lose all dataset/model messages |

⚠️ **The body must be in SINGLE quotes.** `$SLURM_PROCID` has to expand *on each node at runtime* — `srun` sets it to 0,1,2,3 for its four tasks. With double quotes the submitting shell expands it once, every node claims rank 0, and the job hangs at the rendezvous forever with no error message. The other variables still resolve inside single quotes because they were `export`ed above and `srun` propagates the environment.

---

## 4. The batch-size arithmetic

The unit of work is a **sequence** of `sequence_length` tokens (2048), packed from documents.

```
global_batch    = per_device_train_batch_size × gradient_accumulation_steps × total_GPUs
tokens_per_step = global_batch × sequence_length
total_tokens    = max_steps × tokens_per_step
```

At 4 nodes: `1 × 2 × 16 = 32` sequences → `32 × 2048 = 65,536` tokens per optimizer step. A 1.3B-token run needs ~19,840 steps.

**Why global batch is the number that matters.** It sets how noisy each gradient estimate is, and therefore what learning rate is appropriate. **Two runs with different global batches are different experiments**, even if everything else matches. Decide it once, hold it for the whole thesis.

**The knobs are not interchangeable:**

- **`per_device_train_batch_size`** costs **memory** — activations scale with it. Cheapest speed win *if* you have memory spare. You don't at 1 node (~56 of 64 GB), and you don't need it at 4 (utilization already 86–99%).
- **`gradient_accumulation_steps`** costs **no memory**. Each GPU does N forward/backward passes, summing gradients, before the optimizer updates. This is the free knob for holding global batch fixed as GPU count changes.
- **`total_GPUs`** is what you're varying.

**Procedure:** pick the global batch (32), pick `per_device` from what memory allows (1), then `grad_accum = global_batch ÷ (per_device × total_GPUs)`.

| Nodes | GPUs | per_device | grad_accum | global batch |
|---|---|---|---|---|
| 1 | 4 | 1 | 8 | 32 |
| 2 | 8 | 1 | 4 | 32 |
| 4 | 16 | 1 | 2 | 32 |
| 8 | 32 | 1 | 1 | 32 |

**8 nodes is the ceiling** — `grad_accum` cannot go below 1. Beyond that the global batch must grow and the learning rate be retuned.

**Verify it every run:** `train_samples_per_second × train_runtime` should equal `max_steps × global_batch`. At 20 steps × 32 that is 640, at every node count. This is what caught the broken multi-node runs.

---

## 5. Every export, and what breaks without it

### Paths

```bash
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
```
Shorthand only.

### HuggingFace

```bash
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
```

**`HF_HOME`** — where models and datasets cache. Must be `$WORK`; `$HOME` has a 50 GB quota and one AFM download is 9.24 GB. Set **explicitly here** because a SLURM batch script is a *non-interactive* shell and reads neither `~/.bashrc` nor `~/.bash_profile`.

**`HF_HUB_OFFLINE=1`** — forbids any network call. **Compute nodes have no internet.** Without this, HuggingFace tries to contact the hub to check for updates, blocks, retries, and either hangs the job or crashes it. With it, only the local cache is used — which is why the cache must be *complete* before you submit. An incomplete snapshot fails hard here rather than silently re-downloading.

**Not set: `TRANSFORMERS_CACHE`.** Deprecated in favour of `HF_HOME`; at best it does nothing.

### JIT compilation caches

```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}_${SLURM_PROCID}
```

Triton (which compiles the KDA kernels) and DeepSpeed both compile code at runtime and cache the result. Their defaults are `~/.triton` and `~/.cache/torch_extensions` — on a **shared filesystem**. With 16 processes compiling into one directory simultaneously you get races: one process reads a half-written file another is still producing. The symptoms look like corruption or missing symbols and are maddening to diagnose.

`/tmp` is **node-local**, and adding job id + rank gives every process its own directory. Set inside the `srun` body so `$SLURM_PROCID` resolves per node.

⚠️ `/tmp` on Booster is RAM-backed and capped at 10 GB. Fine for JIT caches (tens of MB). Never write large intermediates there — it eats node RAM and OOMs the job confusingly.

### Weights & Biases

```bash
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
```

**Why offline:** wandb normally streams metrics to their servers in real time. Compute nodes have no internet, so it would block on connect, retry, and either hang the job or flood the log with errors. Offline mode writes everything locally instead.

Afterwards, from a **login** node (which does have internet):

```bash
wandb sync $WORK_LUKAS/wandb/wandb/offline-run-*
```

**`WANDB_DIR`** puts those files on `$WORK` rather than the 50 GB `$HOME`.

Related: `report_to` defaults to `"all"` in HF `TrainingArguments`, which means wandb whenever it is installed — and it is. So this is not optional unless you set `report_to: none` in the YAML.

### Threading

```bash
export OMP_NUM_THREADS=8       # 32 cores ÷ 4 processes per node
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
```

**`OMP_NUM_THREADS`** — without it, OpenMP defaults to "all cores" *per process*. Four processes × 32 threads = 128 threads contending for 32 cores. Everything gets slower. `MKL_NUM_THREADS` is the same for Intel's math library.

**`TOKENIZERS_PARALLELISM=false`** — HuggingFace's fast tokenizers use Rust threads, which is unsafe once the process forks (dataloader workers, `dataset_num_proc`). HF prints a warning on every fork. Setting this silences it; the parallelism you actually want comes from `dataset_num_proc` in the YAML.

### TF32

```bash
export TORCH_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

Inherited from the practical-work sbatch. The intent is to let matmuls use TF32 on Ampere — faster, slightly less precise. **Honest caveat: I am not certain these are variables PyTorch actually reads** (the documented routes are `torch.backends.cuda.matmul.allow_tf32` in code, or `NVIDIA_TF32_OVERRIDE`). They are harmless either way, and largely moot for bf16 training where the matmuls are already bf16.

### GPU monitor

```bash
(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || echo "$(date) NVML_ERROR"
    sleep 5
  done
) > $WORK_LUKAS/distillkit/logs/gpu_${SLURM_JOB_ID}_$(hostname).log &
```

Background loop sampling every 5 seconds. **`sleep 5`, not 30** — a ten-second run is essentially invisible at 30 s intervals. Runs on **node 0 only** (it is before `srun`); `$(hostname)` in the filename matters only if you move it inside `srun`.

The Trainer's own `skip_memory_metrics: false` numbers are more precise anyway — `train_mem_gpu_alloc_delta` and `train_mem_gpu_peaked_delta` measure allocated tensors directly, whereas nvidia-smi also counts the allocator's reserved-but-free blocks, the CUDA context and NCCL buffers (which is why nvidia-smi reads ~56 GB where the Trainer accounts for ~44 GB).

---

## 6. Quick reference: what to check after every multi-node run

```bash
J=<jobid>; F=logs/<name>-$J.out
grep -c "train_runtime" $F                      # MUST be 1 (one world, not N)
grep -o "nranks [0-9]*" $F | sort -u            # MUST equal total GPUs
grep -o "nNodes [0-9]*"  $F | sort -u           # MUST equal node count
grep -o "train_mem_gpu_alloc_delta': [0-9]*" $F # MUST follow 9.25 + 55.46/W GB
grep -c "eval_loss" $F                          # 0 for timing runs
grep -i "NET/IB" $F | head -1                   # InfiniBand, not NET/Socket
```

`train_mem_gpu_alloc_delta` is the physical invariant: 23.1 GB at 4 GPUs, 16.2 at 8, 12.7 at 16. If it does not move with node count, the job did not merge.