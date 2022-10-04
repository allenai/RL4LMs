# RL4LMs - A modular RL library to fine-tune language models for natural language generation tasks.

We provide easily customizable building blocks for training language models including implementations of on-policy algorithms, reward functions, metrics, datasets and LM actor-critic policies

Thoroughly tested and benchmarked on a comprehensive set of: 
- 6 different Natural Language Processing (NLP) Tasks:
    - Summarization
    - Generative Commonsense Reasoning
    - IMDB Sentiment-based Text Continuation
    - Table-to-text generation
    - Abstractive Question Answering
    - Machine Translation
- Different types of NLG metrics which can be used reward functions:
    - Lexical Metrics (ROUGE, BLEU, SacreBLEU, METEOR)
    - Semantic Metrics (BERTSCORE, BLEURT)
    - Task specific metrics ()
- On-policy algorithms of PPO, A2C, TRPO and novel **NLPO (Natural Language Policy Optimization)**
- Actor-Critic Policies supporting causal LMs (eg. GPT-2/3) and seq2seq LMs (eg. T5, BART)

---
# Install

## Local Installation 
```bash
git clone https://github.com/allenai/RL4LMs.git
cd RL4LMs
pip install -e .
```

## Docker
We provide also a Dockerfile for development using docker containers containing all the dependencies.
```bash
docker build -t . rl4lms
```

## Additional dependencies

Optionally, coreNLP libraries are required for certain metric computations (eg. SPICE) which can be downloaded using the bash script `rl4lms/envs/text_generation/caption_metrics/spice`

---
# Quick Start - Train PPO/NLPO using pre-defined YAML configs
We provide a simple training API that can be invoked via `scripts/training/train_text_generation.py` that allows to train PPO, NLPO or a supervised model by using a config file (YAML). 

For example, to train T5-base on CNN/DM summarization on PPO using Rouge-1 as reward function, you can run:

```bash
python scripts/train_text_generation.py --config_path scripts/task_gen_configs/summarization/t5_ppo.yml
```

## YAML file schema - Configuring building blocks

Config file contains details about hyper-parameter settings for building blocks and they are described below:

- **Dataset/Task**: Dataset containing samples with input prompts and reference sentences. Available datasets are found in the class `DataPoolRegistry` in  `rl4lms/envs/text_generation/registry.py`. (See how to create your own dataset below)

  ```yaml
  # dataset
  datapool:
    id: cnn_daily_mail
    args:
      prompt_prefix: "Summarize: "
  ```

- **Tokenizer** - A pre-trained tokenizer that is used to (de)tokenize input and output sequences with settings for padding and truncation
  ```yaml
  # tokenizer
  tokenizer:
    model_name: t5-base
    padding_side: left
    truncation_side: left
    pad_token_as_eos_token: False

  ``` 
- **Reward Function**: Reward function which computes token-level scores at each time step of MDP, that is configured using ID and arguments. Available reward functions can be found in the class `RewardFunctionRegistry` in  `rl4lms/envs/text_generation/registry.py`. (See how to create your own reward function below)

  ```yaml
  # reward function that is optimized
  reward_fn:
  id: rouge
  args:
    rouge_type: "rouge1"
  ```

- **Environment**: Configures a gym-style environment `rl4lms/envs/text_generation/env.py` which simulates and generates MDP episodes. We use vectorized environment from stable-baselines that processes `n_envs` episodes in parallel using multi-processing to compute step-wise rewards.  Further parameters that can be configured are: `max_episode_length` - max length of the episode, `max_prompt_length` - maximum length of the input text to consider, `terminate_on_eos` - whether to terminate the episode as soon as EOS action is performed, `prompt_truncation_side` - truncation side for the prompt text, `context_start_token` - token id for context token (corresponds to initial token given to decoder in encoder-decoder models)

  ```yaml
  env:
    n_envs: 10
    args:
      max_prompt_length: 512
      max_episode_length: 100
      terminate_on_eos: True
      prompt_truncation_side: "right"
      context_start_token: 0
  ```

- **On-policy alg**: TBD
    ```yaml
    alg:
      id: ppo
      args: 
        n_steps: 512
        batch_size: 64
        verbose: 1
        learning_rate: 0.000002
        n_epochs: 5
        ent_coef: 0.0
      kl_div:
        coeff: 0.001
        target_kl: 0.2
      policy:
        id: seq2seq_lm_actor_critic_policy
        args:
          model_name: t5-base
          apply_model_parallel: True
          prompt_truncation_side: "right"
          generation_kwargs:
            do_sample: True
            top_k: 50
            min_length: 50
            max_new_tokens: 100          
    ```

- **Trainer Config**: TBD

  ```yaml
  # train and evaluation
  train_evaluation:
    eval_batch_size: 100
    n_iters: 100
    eval_every: 10
    save_every: 1
    metrics:
      - id: meteor
        args: {}
      - id: rouge
      - id: bleu
        args: {}
      - id: bert_score
        args:
          language: en
      - id: diversity
        args: {}
    generation_kwargs: 
      do_sample: True
      top_k: 0
      temperature: 0.7
      min_length: 50
      max_new_tokens: 100
  ```

## Configs
Configs for training other tasks and algorithms can be found in: `scripts/training/task_configs`

---
# Custom Components (TBD)
RL4LMs provide full customizability - with respect to adding new tasks/datasets, reward functions, evaluation metrics and actor-critic policies.


# Custom trainer (training loop) (TBD)

---

# Logging

Additionally, we support WANDB logging and warm-starting of training by storing checkpoints and other training artifacts in a user-specified path. This is especially useful for running preemptible jobs on larged, scheduled clusters.
```bash 
WANDB_API_KEY=<YOUR-WANDB-API-KEY-HERE>  python scripts/training/train_text_generation.py --config_path <PATH-TO-CONFIG-FILE> --experiment_name <EXPERIMENT-NAME> --base_path_to_store_results <PATH-TO-STORE-RESULTS> --log_to_wandb