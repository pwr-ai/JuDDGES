# Running fine-tuning on SLURM cluster
Runs `scripts/sft/fine_tune_deepspeed.py` on SLURM cluster with a given model and dataset.

> [!NOTE]
> We currently support only single-node, multi-gpu training.

## Instructions

1. Create a .env file with the variables exported at the top of the `run_on_cluster.sh` script.
2. Build .sif image using dockerfile in `slurm/fine_tuning_env.dockerfile` and make sure it'll be available on the cluster (after running sbatch)
3. Submit a job to the cluster using `sbatch` (adjust allocation parameters as needed by overriding them in CLI args right after `sbatch` call, as `--job-name` below)
    ```bash
    sbatch \
        --job-name sft
        run_on_cluster.sh \
            --model llama_3.1_8b_instruct \
            --dataset pl-frankowe-instruct
    ```
