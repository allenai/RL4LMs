from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecEnv
from transformers import PreTrainedTokenizer
from accelerate import Accelerator

from rl4lms.algorithms.common.maskable.buffers import MaskableDictRolloutBuffer
from rl4lms.envs.text_generation.kl_controllers import KLController
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.policy.base_policy import (
    PolicyOutput,
    RefPolicyOutput,
    ValueOutput,
)
from rl4lms.envs.text_generation.reward import BatchedRewardFunction, RewardFunction
from rl4lms.envs.text_generation.warm_start import OnPolicyWarmStartMixin


@dataclass
class TransitionInfo:
    observation: TensorDict
    action: np.ndarray
    task_reward: np.ndarray
    total_reward: np.ndarray
    kl_div: np.ndarray
    episode_start: np.ndarray
    value: torch.Tensor
    log_prob: torch.Tensor
    done: np.ndarray
    ref_log_prob: torch.Tensor
    kl_reward: np.ndarray
    action_mask: np.ndarray
    info: Dict[str, Any]


def unpack_observations(obs_tensor, n_envs: int):
    """
    Unpacks vectorized dict observations into separate dict observations
    """
    unpacked_obs = []
    keys = obs_tensor.keys()
    for env_ix in range(n_envs):
        obs_dict = {}
        for key in keys:
            obs_dict[key] = obs_tensor[key][env_ix].reshape(1, -1).cpu()
        unpacked_obs.append(obs_dict)
    return unpacked_obs


def compute_batched_rewards(
    episode_wise_transitions: List[List[TransitionInfo]], reward_fn: RewardFunction
):
    # first collect all the prompts, ref and gen texts
    prompts = []
    reference_texts = []
    generated_texts = []
    is_dones = []
    indices = []
    meta_infos = []
    for env_ix, transitions in enumerate(episode_wise_transitions):
        for trans_ix, transition in enumerate(transitions):
            done = transition.done
            info = transition.info
            prompts.append(info["prompt_text"])
            reference_texts.append(info["reference_text"])
            generated_texts.append(info["output"])
            is_dones.append(done)
            meta_infos.append(info["meta_info"])
            indices.append((env_ix, trans_ix))

    # compute rewards all at once
    rewards = reward_fn(prompts, generated_texts, reference_texts, is_dones, meta_infos)
    # rewards = rewards.numpy().flatten()

    # override the rewards in transitions
    for (env_ix, trans_ix), reward in zip(indices, rewards):
        episode_wise_transitions[env_ix][trans_ix].task_reward = reward
        episode_wise_transitions[env_ix][trans_ix].total_reward = (
            reward + episode_wise_transitions[env_ix][trans_ix].kl_reward
        )


def wrap_onpolicy_alg(
    alg_class: Type[OnPolicyAlgorithm],
    alg_kwargs: Dict[str, Any],
    kl_coeff: float,
    tracker: Tracker,
    accelerator: Accelerator,
    target_kl: float = None,
    norm_reward: bool = False,
):
    class OnPolicyAlgText(alg_class, OnPolicyWarmStartMixin):
        def __init__(
            self,
            alg_kwargs: Dict[str, Any],
            kl_coeff: float,
            tracker: Tracker,
            accelerator: Accelerator,
            target_kl: float = None,
            norm_reward: bool = False,
        ):
            alg_kwargs["tracker"] = tracker
            alg_kwargs["accelerator"] = accelerator
            super().__init__(**alg_kwargs)
            self._kl_controller = KLController(kl_coeff, target_kl)
            self.tracker = tracker
            self._norm_reward = norm_reward
            # flattened rollout buffer
            self.rollout_buffer = MaskableDictRolloutBuffer(
                self.n_steps * self.env.num_envs,
                self.observation_space,
                self.action_space,
                device=accelerator.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=1,
            )
            self.reward_fn = self.env.get_attr("reward_function", 0)[0]

        def get_policy_kwargs(
            self,
            obs: TensorDict,
            action: torch.tensor,
            past_state: Dict[str, torch.tensor],
            action_mask: torch.tensor,
        ):

            # send to device
            for key, item in obs.items():
                obs[key] = item.to(self.accelerator.device)

            policy_kwargs = {
                "obs": obs,
                "actions": action.to(self.accelerator.device),
                "past_model_kwargs": past_state,
            }
            if action_mask is not None:
                policy_kwargs["action_masks"] = action_mask.to(self.accelerator.device)
            return policy_kwargs

        def generate_batch(
            self,
            rollout_buffer: DictRolloutBuffer,
            tokenizer: PreTrainedTokenizer,
            max_steps: int,
            rollout_info: Dict[str, Any],
        ):
            # if rollout buffer is already full, do not continue
            if rollout_buffer.full:
                return

            # start parallel episodes
            current_obs = self.env.reset()
            episode_starts = np.ones((self.env.num_envs,), dtype=bool)

            # generate text using the model
            obs_tensor = obs_as_tensor(current_obs, self.device)
            generation_inputs = self.accelerator.unwrap_model(self.policy).get_inputs_for_generation(obs_tensor)
            gen_output = self.accelerator.unwrap_model(self.policy).generate(
                input_ids=generation_inputs.inputs,
                attention_mask=generation_inputs.attention_masks,
                tokenizer=tokenizer,
                accelerator=self.accelerator,
                gather_from_devices=False
            )

            # process them one step at a time to collect rollout info
            episode_wise_transitions = [[] for _ in range(self.env.num_envs)]
            ep_terminated = np.zeros((self.env.num_envs,), dtype=bool)
            value_past_state = None
            ref_past_state = None
            policy_past_state = None
            masks = (
                gen_output.action_masks
                if gen_output.action_masks is not None
                else [None] * len(gen_output.step_wise_logprobs)
            )

            for actions_tensor, _, action_mask in zip(
                gen_output.step_wise_actions, gen_output.step_wise_logprobs, masks
            ):
                # if all episodes are done, just break and do not continue
                if np.all(ep_terminated):
                    break

                # evaluate actions with actions from rollout
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(current_obs, self.accelerator.device)

                    # get log probs
                    policy_kwargs = self.get_policy_kwargs(
                        obs_tensor, actions_tensor, policy_past_state, action_mask
                    )

                    policy_outputs: PolicyOutput = self.accelerator.unwrap_model(self.policy).forward_policy(
                        **policy_kwargs
                    )
                    raw_log_probs, log_probs, policy_past_state = (
                        policy_outputs.raw_log_probs,
                        policy_outputs.log_probs,
                        policy_outputs.past_model_kwargs,
                    )

                    # sanity check
                    assert torch.all(
                        torch.isfinite(log_probs)
                    ), "Infinite values in log probs"

                    # sanity check
                    assert torch.all(
                        torch.isfinite(raw_log_probs)
                    ), "Infinite values in log probs"

                    # get values
                    value_outputs: ValueOutput = self.accelerator.unwrap_model(self.policy).forward_value(
                        obs_tensor, value_past_state
                    )
                    values, value_past_state = (
                        value_outputs.values,
                        value_outputs.past_model_kwargs,
                    )

                    # get reference log probs
                    ref_policy_outputs: RefPolicyOutput = (
                        self.accelerator.unwrap_model(self.policy).get_log_probs_ref_model(
                            obs_tensor, actions_tensor, ref_past_state
                        )
                    )
                    ref_log_probs, ref_past_state = (
                        ref_policy_outputs.log_probs,
                        ref_policy_outputs.past_model_kwargs,
                    )

                    # sanity check
                    assert torch.all(
                        torch.isfinite(ref_log_probs)
                    ), "Infinite values in log probs"

                    # compute KL rewards
                    kl_div = raw_log_probs - ref_log_probs
                    kl_rewards = -1 * self._kl_controller.kl_coeff * kl_div

                # step into env to get rewards
                actions = actions_tensor.cpu().numpy()
                new_obs, rewards, dones, infos = self.env.step(actions)

                self.num_timesteps += self.env.num_envs

                # compute total rewards
                total_rewards = rewards + kl_rewards.cpu().numpy()

                # unpack individual observations
                unpacked_obs = unpack_observations(obs_tensor, self.env.num_envs)

                # store episode wise transitions separately
                for env_ix in range(self.env.num_envs):
                    # only if not terminated already
                    if not ep_terminated[env_ix]:
                        transtion = TransitionInfo(
                            observation=unpacked_obs[env_ix],
                            action=actions[env_ix],
                            task_reward=rewards[env_ix],
                            total_reward=total_rewards[env_ix],
                            kl_div=kl_div.cpu().numpy()[env_ix],
                            episode_start=episode_starts[env_ix],
                            value=values[env_ix].cpu(),
                            log_prob=log_probs[env_ix].cpu(),
                            done=dones[env_ix],
                            ref_log_prob=ref_log_probs[env_ix].cpu(),
                            kl_reward=kl_rewards.cpu().numpy()[env_ix],
                            action_mask=action_mask[env_ix].cpu().numpy()
                            if action_mask is not None
                            else None,
                            info=infos[env_ix],
                        )

                        episode_wise_transitions[env_ix].append(transtion)

                    # mark this episode to terminated if done occurs once
                    if dones[env_ix]:
                        ep_terminated[env_ix] = True

                episode_starts = np.zeros((self.env.num_envs,), dtype=bool)
                current_obs = new_obs

            # now we flush all episode wise info to the 1-D buffer
            rollout_info = self._add_to_buffer(
                rollout_buffer, episode_wise_transitions, rollout_info
            )
            return rollout_info

        def _add_to_buffer(
            self, rollout_buffer, episode_wise_transitions, rollout_info
        ):
            # if the reward function is batchable, we override the rewards here
            if isinstance(self.reward_fn, BatchedRewardFunction):
                compute_batched_rewards(episode_wise_transitions, self.reward_fn)

            advantages_computed = False
            for ep_ix, transitions in enumerate(episode_wise_transitions):
                ep_length = len(transitions)
                total_reward = 0.0
                total_kl_reward = 0.0
                for transition_ix, transition in enumerate(transitions):
                    total_reward += transition.task_reward
                    total_kl_reward += transition.kl_reward
                    rollout_info["rollout_info/kl_div_mean"].append(transition.kl_div)
                    rollout_info["rollout_info/log_prob"].append(transition.log_prob)
                    rollout_info["rollout_info/ref_log_prob"].append(
                        transition.ref_log_prob
                    )
                    rollout_info["rollout_info/values"].append(transition.value.numpy())

                    if not rollout_buffer.full:
                        rollout_buffer.add(
                            transition.observation,
                            transition.action,
                            transition.total_reward,
                            transition.episode_start,
                            transition.value,
                            transition.log_prob,
                            action_masks=transition.action_mask,
                        )

                    # if the buffer is full, compute advantages
                    if rollout_buffer.full and not advantages_computed:

                        # normalize the rewards
                        if self._norm_reward:
                            mean = rollout_buffer.rewards.mean()
                            std = rollout_buffer.rewards.std()
                            rollout_buffer.rewards = (rollout_buffer.rewards - mean) / (
                                std + 1e-8
                            )

                        # we fetch the last value for the last time step
                        # values come from the next transitions's values
                        next_values = (
                            transitions[transition_ix + 1].value
                            if (transition_ix + 1) < ep_length
                            else torch.tensor([0.0])
                        )

                        rollout_buffer.compute_returns_and_advantage(
                            last_values=next_values, dones=transition.done
                        )
                        advantages_computed = True

                rollout_info["rollout_info/ep_rew"].append(total_reward)
                rollout_info["rollout_info/ep_lens"].append(ep_length)
                rollout_info["rollout_info/ep_kl_rew"].append(total_kl_reward)
            return rollout_info

        def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
        ) -> bool:
            # max episode steps
            max_steps = env.unwrapped.get_attr("max_steps", [0])[0]

            # get tokenizer
            tokenizer = env.unwrapped.get_attr("tokenizer", [0])
            tokenizer = tokenizer[0]

            # Switch to eval mode
            self.accelerator.unwrap_model(self.policy).set_training_mode(False)

            # reset rollout buffer and stats
            rollout_buffer.reset()

            # start the rollout process
            rollout_info = {
                "rollout_info/ep_rew": [],
                "rollout_info/kl_div_mean": [],
                "rollout_info/ep_lens": [],
                "rollout_info/ep_kl_rew": [],
                "rollout_info/log_prob": [],
                "rollout_info/ref_log_prob": [],
                "rollout_info/values": [],
            }
            while not rollout_buffer.full:
                # generate batch of rollouts
                rollout_info = self.generate_batch(
                    rollout_buffer, tokenizer, max_steps, rollout_info
                )

            # aggregate rollout info
            aggregated_rollout_info = {}
            for key, values in rollout_info.items():
                aggregated_rollout_info[key] = np.mean(values).item()
                aggregated_rollout_info[f"{key}_std"] = np.std(values).item()
            aggregated_rollout_info[
                "rollout_info/kl_coeff"
            ] = self._kl_controller.kl_coeff

            # TBD - gather rollout stats 

            if self.tracker is not None:
                self.tracker.log_rollout_infos(aggregated_rollout_info)

            # adapt the KL coeff
            self._kl_controller.step(
                torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"])
            )
            return True

    # instantiate the wrapped alg
    alg = OnPolicyAlgText(alg_kwargs, kl_coeff, tracker, accelerator, target_kl, norm_reward)
    return alg
