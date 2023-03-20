from functools import partial
from typing import Any, Dict, List
import numpy as np
import torch

from rl4lms.data_pools.text_generation_pool import Sample
from rl4lms.envs.text_generation.env import TextGenEnv
from rl4lms.envs.text_generation.evaluation_utils import evaluate_on_samples
from rl4lms.envs.text_generation.utils_supervised import (
    evaluate_on_samples as evaluate_supervised,
)
from rl4lms.envs.text_generation.dataset_utils import create_dataloader
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.registry import (
    DataPoolRegistry,
    MetricRegistry,
    RewardFunctionRegistry,
    PolicyRegistry,
    AlgorithmRegistry,
    WrapperRegistry,
)
from rl4lms.envs.text_generation.reward import RewardFunction
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from accelerate import Accelerator

from rl4lms.envs.text_generation.utils_supervised import (
    get_datasets_for_causal,
    get_datasets_for_seq2seq,
    tokenize_causal,
    tokenize_seq2seq,
    EvalCallack,
)
from rl4lms.envs.text_generation.warm_start import TrainerWarmStartMixin
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from transformers import LlamaConfig, LlamaForCausalLM

def fsdp_prepare(policy, env, device):
    """
    Prepare models for distributed setup
    especially for FSDP related issues
    https://github.com/huggingface/accelerate/issues/947#event-8448457764
    """
    with torch.no_grad():
        obs = env.reset()
        obs_tensor = obs_as_tensor(obs, device)
        outputs = policy(obs_tensor, actions=None)


def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(
        reward_config["id"], reward_config.get("args", {})
    )
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [
        MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
        for metric_config in metric_configs
    ]
    return metrics


def build_datapool(datapool_config: Dict[str, Any]):
    def _get_datapool_by_split(split: str):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train")
    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": [(sample, weight) for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool],
    }
    return samples_by_split


def build_env(
    env_config: Dict[str, Any],
    reward_fn: RewardFunction,
    tokenizer: AutoTokenizer,
    train_samples: List[Sample],
):
    # vectoried env
    env_kwargs = {
        "reward_function": reward_fn,
        "tokenizer": tokenizer,
        "samples": train_samples,
    }
    env_kwargs = {**env_kwargs, **env_config.get("args", {})}
    env = make_vec_env(
        TextGenEnv,
        n_envs=env_config.get("n_envs", 1),
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
    )
    return env


def build_alg(
    alg_config: Dict[str, Any],
    env: TextGenEnv,
    tracker: Tracker,
    accelerator: Accelerator,
    policy_state: Dict[str, Any],
    alg_state: Dict[str, Any],
):
    policy_config = alg_config["policy"]
    policy_cls = PolicyRegistry.get(policy_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])

    policy_args = policy_config["args"]
    policy_args["state_dict"] = policy_state
    policy_args["dist_type"] = accelerator.distributed_type
    alg_kwargs = {
        "policy": policy_cls,
        "env": env,
        "policy_kwargs": policy_args,
    }
    alg_kwargs = {**alg_kwargs, **alg_config.get("args")}
    wrapper = WrapperRegistry.get(alg_config["id"])
    alg = wrapper(
        alg_cls,
        alg_kwargs,
        alg_config["kl_div"]["coeff"],
        tracker,
        accelerator,
        alg_config["kl_div"].get("target_kl", None),
        alg_config["kl_div"].get("norm_reward", False),
    )
    alg.load_from_dict(alg_state)
    return alg


class OnPolicyTrainer(TrainerWarmStartMixin):
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        datapool_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        env_config: Dict[str, Any],
        on_policy_alg_config: Dict[str, Any],
        train_eval_config: Dict[str, Any],
        accelerator: Accelerator,
        tracker: Tracker = None,
        experiment_name: str = "",
    ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._on_policy_alg_config = on_policy_alg_config
        self._train_eval_config = train_eval_config
        self._accelerator = accelerator
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._setup()

    def _setup(self):
        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(self._datapool_config)
        self._env = build_env(
            self._env_config,
            self._reward_fn,
            self._tokenizer,
            self._samples_by_split["train"],
        )
        self._alg = build_alg(
            self._on_policy_alg_config,
            self._env,
            self._tracker,
            self._accelerator,
            self._policy_state_dict,
            self._alg_state_dict,
        )

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._n_steps_per_iter = self._env.num_envs * self._alg.n_steps

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get("generation_kwargs", None)

        # prepare for accelerate
        self._prepare_accelerate()

    def _prepare_accelerate(self):
        self._alg.policy = self._accelerator.prepare(self._alg.policy)
        optimizer = self._alg.policy.setup_optimizer()

        # prepare dataloaders
        self._dataloaders = {
            "val": create_dataloader(self._samples_by_split["val"], self._eval_batch_size),
            "test": create_dataloader(self._samples_by_split["test"], self._eval_batch_size),
        }

        # prepare policy, optimizer and dataloader
        (
            self._alg.optimizer,
            self._dataloaders["val"],
            self._dataloaders["test"],
        ) = self._accelerator.prepare(optimizer,
                                     self._dataloaders["val"],
                                     self._dataloaders["test"])




    def _evaluate_on_datapools(self, epoch: int, splits: List[str] = ["val", "test"]):
        # need to do this for FSDP
        fsdp_prepare(self._alg.policy, self._env, self._accelerator.device)

        for split in splits:
            evaluate_on_samples(
                policy=self._alg.policy,
                tokenizer=self._tokenizer,
                dataloader=self._dataloaders[split],
                max_prompt_length=self._max_prompt_length,
                metrics=self._metrics,
                epoch=epoch,
                split_name=split,
                accelerator=self._accelerator,
                tracker=self._tracker,
                gen_kwargs=self._eval_gen_kwargs,
            )

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        #self._evaluate_on_datapools(epoch=iter_start)

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            # save the policy checkpoint
            if (epoch + 1) % self._train_eval_config.get("save_every", 20) == 0:
                self.save_trainer_state(
                    self._tracker, self._accelerator.unwrap_model(self._alg.policy), self._trainer_state)

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                self._evaluate_on_datapools(epoch=epoch, splits=["val"])

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch)

        # save model here - we save only the language model
        if self._tracker is not None:

            # wait for every process
            self._accelerator.wait_for_everyone()

            self._tracker.save_auto_model(
                self._accelerator.unwrap_model(self._alg.policy).get_language_model())


class SupervisedTrainer:
    """
    A supervised trainer to train LMs (causal and seq2seq) on text generation tasks (wrapper on HF trainer)
    """

    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        datapool_config: Dict[str, Any],
        train_eval_config: Dict[str, Any],
        alg_config: Dict[str, Any],
        tracker: Tracker = None,
        accelerator: Accelerator = None
    ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._train_eval_config = train_eval_config
        self._alg_config = alg_config
        self._tracker = tracker
        self._accelerator = accelerator
        self._setup()

    def _evaluate_on_datapools(self, epoch: int, splits: List[str] = ["val", "test"]):
        # also prepare model?
        #self._model = self._accelerator.prepare(self._model)
        batch = self._tokenizer(["random text"], return_tensors="pt").to(self._accelerator.device)
        outputs = self._model(**batch)

        for split in splits:
            evaluate_supervised(
                model=self._model,
                tokenizer=self._tokenizer,
                dataloader=self._dataloaders[split],
                max_prompt_length=self._max_prompt_length,
                metrics=self._metrics,
                epoch=epoch,
                split_name=split,
                tracker=self._tracker,
                generation_kwargs=self._gen_kwargs,
                accelerator=self._accelerator
            )
        
        # wait for everyone
        self._accelerator.wait_for_everyone()

    def _setup(self):
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._metrics = build_metrics(self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(self._datapool_config)
        self._train_dataset = (
            get_datasets_for_causal(self._samples_by_split["train"])
            if self._alg_config["model_type"] == "causal"
            else get_datasets_for_seq2seq(self._samples_by_split["train"])
        )
        preprocess_fn = (
            tokenize_causal
            if self._alg_config["model_type"] == "causal"
            else tokenize_seq2seq
        )
        preprocess_fn = partial(preprocess_fn, tokenizer=self._tokenizer)
        with self._accelerator.local_main_process_first():
            self._tokenized_dataset = self._train_dataset.map(
                preprocess_fn, batched=True,remove_columns=self._train_dataset.column_names
        )
        self._model_cls = (
            AutoModelForCausalLM
            if self._alg_config["model_type"] == "causal"
            else AutoModelForSeq2SeqLM
        )
        self._gen_kwargs = self._alg_config["generation_kwargs"]

        #device_map = {'model.embed_tokens': 'cpu', 'model.layers.0': 'cpu', 'model.layers.1': 'cpu', 'model.layers.2': 'cpu', 'model.layers.3': 'cpu', 'model.layers.4': 'cpu', 'model.layers.5': 'cpu', 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.layers.32': 'cpu', 'model.layers.33': 'cpu', 'model.layers.34.self_attn.q_proj': 'cpu', 'model.layers.34.self_attn.k_proj': 'cpu', 'model.layers.34.self_attn.v_proj': 'cpu', 'model.layers.34.self_attn.o_proj': 'cpu', 'model.layers.34.self_attn.rotary_emb': 'cpu', 'model.layers.34.mlp': 'cpu', 'model.layers.34.input_layernorm': 'cpu', 'model.layers.34.post_attention_layernorm': 'cpu', 'model.layers.35': 'cpu', 'model.layers.36': 'cpu', 'model.layers.37': 'cpu', 'model.layers.38': 'cpu', 'model.layers.39': 'cpu', 'model.layers.40': 'cpu', 'model.layers.41': 'cpu', 'model.layers.42': 'cpu', 'model.layers.43': 'cpu', 'model.layers.44': 'cpu', 'model.layers.45': 'cpu', 'model.layers.46': 'cpu', 'model.layers.47': 'cpu', 'model.layers.48': 'cpu', 'model.layers.49': 'cpu', 'model.layers.50': 'cpu', 'model.layers.51': 'cpu', 'model.layers.52': 'cpu', 'model.layers.53': 'cpu', 'model.layers.54': 'cpu', 'model.layers.55': 'cpu', 'model.layers.56': 'cpu', 'model.layers.57': 'cpu', 'model.layers.58': 'cpu', 'model.layers.59': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}

        #device_map = {'model.embed_tokens': 0, 'model.layers.0': 1, 'model.layers.1': 2, 'model.layers.2': 3, 'model.layers.3': 4, 'model.layers.4': 5, 'model.layers.5': 6, 'model.layers.6': 7, 'model.layers.7': 8, 'model.layers.8': 9, 'model.layers.9': 10, 'model.layers.10': 11, 'model.layers.11': 12, 'model.layers.12': 13, 'model.layers.13': 14, 'model.layers.14': 15, 'model.layers.15': 16, 'model.layers.16': 17, 'model.layers.17': 18, 'model.layers.18': 19, 'model.layers.19': 20, 'model.layers.20': 21, 'model.layers.21': 22, 'model.layers.22': 23, 'model.layers.23': 24, 'model.layers.24': 25, 'model.layers.25': 26, 'model.layers.26': 27, 'model.layers.27': 28, 'model.layers.28': 29, 'model.layers.29': 30, 'model.layers.30': 31, 'model.layers.31': 32, 'model.layers.32': 33, 'model.layers.33': 34, 'model.layers.34.self_attn.q_proj': 35, 'model.layers.34.self_attn.k_proj': 36, 'model.layers.34.self_attn.v_proj': 37, 'model.layers.34.self_attn.o_proj': 38, 'model.layers.34.self_attn.rotary_emb': 39, 'model.layers.34.mlp': 0, 'model.layers.34.input_layernorm': 1, 'model.layers.34.post_attention_layernorm': 2, 'model.layers.35': 3, 'model.layers.36': 4, 'model.layers.37': 5, 'model.layers.38': 6, 'model.layers.39': 7, 'model.layers.40': 8, 'model.layers.41': 9, 'model.layers.42': 10, 'model.layers.43': 11, 'model.layers.44': 12, 'model.layers.45': 13, 'model.layers.46': 14, 'model.layers.47': 15, 'model.layers.48': 16, 'model.layers.49': 17, 'model.layers.50': 18, 'model.layers.51': 19, 'model.layers.52': 20, 'model.layers.53': 21, 'model.layers.54': 22, 'model.layers.55': 23, 'model.layers.56': 24, 'model.layers.57': 25, 'model.layers.58': 26, 'model.layers.59': 27, 'model.norm': 28, 'lm_head': 29}

        #device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0, 'model.layers.32.self_attn': 0, 'model.layers.32.mlp.gate_proj': 0, 'model.layers.32.mlp.down_proj': 0, 'model.layers.32.mlp.up_proj': 1, 'model.layers.32.mlp.act_fn': 1, 'model.layers.32.input_layernorm': 1, 'model.layers.32.post_attention_layernorm': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1, 'model.layers.37': 1, 'model.layers.38': 1, 'model.layers.39': 1, 'model.norm': 1, 'lm_head': 1}

        # new_device_map = {}
        # world_size = torch.distributed.get_world_size() #40
        # local_world_size = torch.cuda.device_count() # 8
        # machines = int(world_size / local_world_size)
        # rank = torch.distributed.get_rank()
        #
        # for i, k in enumerate(device_map.keys()):
        #     layer_device_map_start = int(rank / local_world_size) * local_world_size
        #     layer_idx = i % local_world_size
        #     new_device_map[k] = 'cpu'#''layer_device_map_start + layer_idx
        #
        # print("NEW DEVICE MAP", world_size, local_world_size, rank, new_device_map)

        #if self._accelerator.is_main_process:
        #memory_dict = {i: '2GIB' for i in range(40)}
        #memory_dict['cpu'] = '512GIB'
        with self._accelerator.local_main_process_first():
            self._model = self._model_cls.from_pretrained(self._alg_config["model_name"],
                                                          #max_memory=memory_dict,
                                                          low_cpu_mem_usage=True)
            #self._model = self._model_cls.from_pretrained(self._alg_config["model_name"], offload_folder='.', offload_state_dict=True, low_cpu_mem_usage=True)
            #, no_split_module_classes=["LlamaDecoderLayer"])

#         config = LlamaConfig.from_pretrained(self._alg_config["model_name"])
#         with init_empty_weights():
#             self._model = LlamaForCausalLM(config)
#             map = infer_auto_device_map(self._model)
#             print("DEVICE MAPPPPP", map)
#             print("WORLDDDD", torch.distributed.get_rank()
# , torch.distributed.get_world_size(), torch.cuda.device_count())

    #13b auto device map on one node
    # {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0, 'model.layers.32.self_attn': 0, 'model.layers.32.mlp.gate_proj': 0, 'model.layers.32.mlp.down_proj': 0, 'model.layers.32.mlp.up_proj': 1, 'model.layers.32.mlp.act_fn': 1, 'model.layers.32.input_layernorm': 1, 'model.layers.32.post_attention_layernorm': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1, 'model.layers.37': 1, 'model.layers.38': 1, 'model.layers.39': 1, 'model.norm': 1, 'lm_head': 1}

    #30b auto device map
    # {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0, 'model.layers.32': 0, 'model.layers.33': 0, 'model.layers.34.self_attn.q_proj': 0, 'model.layers.34.self_attn.k_proj': 0, 'model.layers.34.self_attn.v_proj': 0, 'model.layers.34.self_attn.o_proj': 1, 'model.layers.34.self_attn.rotary_emb': 1, 'model.layers.34.mlp': 1, 'model.layers.34.input_layernorm': 1, 'model.layers.34.post_attention_layernorm': 1, 'model.layers.35': 1, 'model.layers.36': 1, 'model.layers.37': 1, 'model.layers.38': 1, 'model.layers.39': 1, 'model.layers.40': 1, 'model.layers.41': 1, 'model.layers.42': 1, 'model.layers.43': 1, 'model.layers.44': 1, 'model.layers.45': 1, 'model.layers.46': 1, 'model.layers.47': 1, 'model.layers.48': 1, 'model.layers.49': 1, 'model.layers.50': 1, 'model.layers.51': 1, 'model.layers.52': 1, 'model.layers.53': 1, 'model.layers.54': 1, 'model.layers.55': 1, 'model.layers.56': 1, 'model.layers.57': 1, 'model.layers.58': 1, 'model.layers.59': 1, 'model.norm': 1, 'lm_head': 1}

        # #self._model = self._model_cls.from_pretrained(self._alg_config["model_name"], offload_folder='.')#, device_map='auto')
        # #self._model = self._model.to('cpu')
        # #self._model = self._accelerator.prepare(self._model)
        # #
        # self._model = load_checkpoint_and_dispatch(
        #     self._model, self._alg_config["model_name"], device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
        # )
        # self._model = self._model.to(self._accelerator.device)

        self._eval_batch_size = self._train_eval_config["eval_batch_size"]

        # setting max prompt length
        self._max_prompt_length = self._tokenizer_config.get(
            "max_length", self._tokenizer.model_max_length
        )

        if (self._alg_config["model_type"] == "causal") and (
            (self._max_prompt_length + self._gen_kwargs["max_new_tokens"])
            > self._tokenizer.model_max_length
        ):
            self._max_prompt_length = (
                self._max_prompt_length - self._gen_kwargs["max_new_tokens"]
            )

        # prepare dataloaders
        self._dataloaders = {
            "val": create_dataloader(self._samples_by_split["val"], self._eval_batch_size),
            "test": create_dataloader(self._samples_by_split["test"], self._eval_batch_size),
        }

        # accelerate prepare
        (
            self._dataloaders["val"],
            self._dataloaders["test"],
        ) = self._accelerator.prepare(self._dataloaders["val"], 
                                     self._dataloaders["test"])

        self._eval_callback = EvalCallack(
            self._dataloaders["val"],
            self._gen_kwargs,
            self._eval_batch_size,
            self._tokenizer,
            self._metrics,
            self._max_prompt_length,
            self._tracker,
            self._accelerator
        )
        train_args = self._alg_config["training_args"]
        train_args["output_dir"] = self._tracker.checkpoint_base_path
        train_args["seed"] = np.random.randint(1e2)  # random seed
        self._train_args = TrainingArguments(**train_args)
        self._data_collator = (
            DataCollatorForLanguageModeling(self._tokenizer, mlm=False)
            if self._alg_config["model_type"] == "causal"
            else DataCollatorForSeq2Seq(self._tokenizer, self._model)
        )

        #if self._accelerator.is_main_process:

    def train_and_eval(self):
        # # evaluate on val and test set before fine-tuning once
        #self._evaluate_on_datapools(epoch=0)

        # might have to reload the model again since trainer does not like the wrapped model
        # self._model = self._model_cls.from_pretrained(self._alg_config["model_name"])

        self._trainer = Trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=self._train_args,
            data_collator=self._data_collator,
            train_dataset=self._tokenized_dataset,
            callbacks=[self._eval_callback],
        )


        # train using HF trainer
        self._trainer.train()

        self._tracker.log_info("TRAINING FINISHED")

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=self._train_args.num_train_epochs)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._accelerator.wait_for_everyone()
            self._tracker.save_auto_model(self._accelerator.unwrap_model(self._model))