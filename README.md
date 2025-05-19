# Fine-tune Whisper on Leonardo


## Helpful commands

<https://slurm.schedmd.com/>

What is the status of all nodes?

```bash
sinfo
```

What is the status of all GPU nodes?

```bash
sinfo -p gpu
```

Start a batch job to SLURM

```bash
sbatch slurm-script.sh
```

What are my processes?

```bash
squeue -u $USER
```

Useful to see if a process is still queueing, running, failing, ...
Even more useful to get the PID for killing.

Kill some process

```bash
scancel PID
```
## Data preparation

The data used to fine-tune Whisper is prepared in `https://github.com/kb-labb/subtitles_preprocessing`
The data is interleaved using the scripts found in  `scripts/interleave/` and executed 
```bash
sbatch --job-name="whisper-interleave" start_script_whisper_interleave.sh scripts/interleave/run_interleave_stage1.sh 
```
## Run Whisper fine-tuning with Huggingface on multiple nodes

Use `start_script_whisper_xxxx.sh` to launch a training job with sbatch, where xxxx is the model size (either tiny, base, small, medium or large). This start script calls one of the training scripts in `scripts/`.

To launch a job, you can issue the following command:

```bash
sbatch --job-name="whisper-large" start_script_whisper_large.sh scripts/whisper_large_multinode.sh 
```

where `whisper_large_multinode.sh` is a positional argument to `start_script_whisper_large.sh`, which should point to one of the training scripts in `scripts/`. 
These scripts are based on the excellent Whisper fine-tuning tutorial from HuggingFace `https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event`
More information on this project as well as the results can be found here: 
`https://kb-labb.github.io/posts/2025-03-07-welcome-KB-Whisper/
`
