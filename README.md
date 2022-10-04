# RL4LMs - A modular RL library to train language models for natural language generation tasks.

We provide building blocks for natural language policy optimization containing on-policy algorithms, reward functions, metrics, datasets and LM actor-critic policies

# Install

## Local Installation 
```bash
git clone https://github.com/allenai/RL4LMs.git
cd RL4LMs
pip install -e .
```

## Docker
We provide also a Dockerfile for development using docker containers containing all the dependencies.


## Additional dependencies

Optionally, coreNLP libraries are required for certain metric computations (eg. SPICE) which can be downloaded using the bash script `rl4lms/envs/text_generation/caption_metrics/spice`

# Quick Start - Train PPO/NLPO using pre-defined YAML configs
We provide a simple training interface `scripts/training/train_text_generation.py` that allows to train PPO, NLPO or supervised by using a config file (YAML). 
For instance to train T5-base on CNN/DM summarization on PPO using Rouge-1 as reward function, one can run:

```bash
python scripts/train_text_generation.py --config_path scripts/text_gen_configs/seq2seq/final_configs/cnn_summarization_ppo.yml
```

Configs for training other tasks and algorithms can be found in: `scripts/text_gen_configs`


Additionally, we support WANDB logging and warm-starting of training by storing checkpoints and other training artifacts in a user-specified path
```bash 
WANDB_API_KEY=<YOUR-WANDB-API-KEY-HERE>  python scripts/training/train_text_generation.py --config_path <PATH-TO-CONFIG-FILE> --experiment_name <EXPERIMENT-NAME> --base_path_to_store_results <PATH-TO-STORE-RESULTS> --log_to_wandb
```

# Train PPO/NLPO using own (TBD)


# Custom Components (TBD)
RL4LMs provide full customizability - with respect to adding new tasks/datasets, reward functions, evaluation metrics and actor-critic policies.


