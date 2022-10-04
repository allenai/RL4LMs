from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
from rl4lms.algorithms.common.maskable.buffers import \
    MaskableDictRolloutBuffer
from rl4lms.envs.text_generation.kl_controllers import KLController
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.reward import (BatchedRewardFunction,
                                                RewardFunction)
from rl4lms.envs.text_generation.warm_start import (OffPolicyWarmStartMixin,
                                                    OnPolicyWarmStartMixin)
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from transformers import PreTrainedTokenizer


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


def compute_batched_rewards(episode_wise_transitions: Dict[str, List[Tuple]],
                            reward_fn: RewardFunction):
    # first collect all the prompts, ref and gen texts
    prompts = []
    reference_texts = []
    generated_texts = []
    is_dones = []
    indices = []
    for env_ix, transitions in enumerate(episode_wise_transitions):
        for trans_ix, transition in enumerate(transitions):
            done = transition[8]
            info = transition[12]
            prompts.append(info["prompt_text"])
            reference_texts.append(info["reference_text"])
            generated_texts.append(info["output"])
            is_dones.append(done)
            indices.append((env_ix, trans_ix))

    # compute rewards all at once
    rewards = reward_fn(
        prompts, generated_texts, reference_texts, is_dones)
    rewards = rewards.numpy().flatten()

    # override the rewards in transitions
    for (env_ix, trans_ix), reward in zip(indices, rewards):
        transition = list(episode_wise_transitions[env_ix][trans_ix])
        transition[2] = reward
        transition[3] = reward + transition[10]
        episode_wise_transitions[env_ix][trans_ix] = tuple(transition)


def wrap_onpolicy_alg(alg_class: Type[OnPolicyAlgorithm],
                      alg_kwargs: Dict[str, Any],
                      kl_coeff: float,
                      tracker: Tracker,
                      target_kl: float = None,
                      norm_reward: bool = False):
    class OnPolicyAlgText(alg_class, OnPolicyWarmStartMixin):
        def __init__(self, alg_kwargs: Dict[str, Any],
                     kl_coeff: float,
                     tracker: Tracker,
                     target_kl: float = None,
                     norm_reward: bool = False):
            alg_kwargs["tracker"] = tracker
            super().__init__(**alg_kwargs)
            self._kl_controller = KLController(kl_coeff, target_kl)
            self.tracker = tracker
            self._norm_reward = norm_reward
            # flattened rollout buffer
            self.rollout_buffer = MaskableDictRolloutBuffer(
                self.n_steps * self.env.num_envs,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=1,
            )
            self.reward_fn = self.env.get_attr("reward_function", 0)[0]

        def generate_batch(self,
                           rollout_buffer: DictRolloutBuffer,
                           tokenizer: PreTrainedTokenizer,
                           max_steps: int,
                           rollout_info: Dict[str, Any]):
            # if rollout buffer is already full, do not continue
            if rollout_buffer.full:
                return

            # start parallel episodes
            current_obs = self.env.reset()
            episode_starts = np.ones((self.env.num_envs,), dtype=bool)

            # generate text using the model
            obs_tensor = obs_as_tensor(current_obs, self.device)
            input_ids, attention_masks = self.policy.get_inputs_for_generation(
                obs_tensor)
            gen_output = self.policy.generate(input_ids=input_ids,
                                              attention_mask=attention_masks,
                                              tokenizer=tokenizer,
                                              )

            # process them one step at a time to collect rollout info
            episode_wise_transitions = [[] for _ in range(self.env.num_envs)]
            ep_terminated = np.zeros((self.env.num_envs,), dtype=bool)
            value_past_state = None
            ref_past_state = None
            policy_past_state = None
            masks = gen_output.get(
                "action_masks", [None] * len(gen_output["step_wise_logprobs"]))
            for actions_tensor, log_probs, action_mask in zip(gen_output["step_wise_actions"],
                                                              gen_output["step_wise_logprobs"],
                                                              masks):

                # sanity check
                assert torch.all(torch.isfinite(log_probs)
                                 ), "Infinite values in log probs"

                # if all episodes are done, just break and do not continue
                if np.all(ep_terminated):
                    break

                # evaluate actions with actions from rollout
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(current_obs, self.device)

                    # # get log probs from policy
                    # _, cache_log_prob, _, _, policy_past_state = self.policy.forward_policy(
                    #     obs_tensor, actions_tensor, policy_past_state)

                    # _, without_cache_log_prob, _, _, policy_past_state = self.policy.forward_policy(
                    #     obs_tensor, actions_tensor, None)
                    # # sanity check 0 - rollout probs and policy probs must match
                    # assert torch.allclose(cache_log_prob, log_probs, atol=1e-3)

                    # # sanity check 1 - log probs with and without cache must match
                    # assert torch.allclose(
                    #     cache_log_prob, without_cache_log_prob, atol=1e-3)

                    # get values
                    values, value_past_state = self.policy.forward_value(obs_tensor,
                                                                         value_past_state)

                    # get reference log probs
                    ref_log_probs, ref_past_state = self.policy.get_log_probs_ref_model(obs_tensor,
                                                                                        actions_tensor,
                                                                                        ref_past_state)

                    # sanity check 2 (this is without caching - must match with values from generate which is with caching)
                    # eval_values, eval_log_probs, _ = self.policy.evaluate_actions(
                    #     obs_tensor, actions_tensor)

                    # assert torch.allclose(
                    #     eval_log_probs, without_cache_log_prob, atol=1e-3)
                    # assert torch.allclose(
                    #     eval_values, values, atol=1e-3)

                    # compute KL rewards
                    kl_div = log_probs - ref_log_probs
                    kl_rewards = -1 * self._kl_controller.kl_coeff * kl_div

                # step into env to get rewards
                actions = actions_tensor.cpu().numpy()
                new_obs, rewards, dones, infos = self.env.step(actions)

                self.num_timesteps += self.env.num_envs

                # compute total rewards
                total_rewards = rewards + kl_rewards.cpu().numpy()

                # unpack individual observations
                unpacked_obs = unpack_observations(
                    obs_tensor, self.env.num_envs)

                # store episode wise transitions separately
                for env_ix in range(self.env.num_envs):
                    # only if not terminated already
                    if not ep_terminated[env_ix]:
                        # TBD: change this DS to dict
                        episode_wise_transitions[env_ix].append(
                            (
                                unpacked_obs[env_ix],             # 0
                                actions[env_ix],                  # 1
                                rewards[env_ix],                  # 2
                                total_rewards[env_ix],            # 3
                                kl_div.cpu().numpy()[env_ix],     # 4
                                episode_starts[env_ix],           # 5
                                values[env_ix].cpu(),             # 6
                                log_probs[env_ix].cpu(),          # 7
                                dones[env_ix],                    # 8
                                ref_log_probs[env_ix].cpu(),      # 9
                                kl_rewards.cpu().numpy()[env_ix],  # 10
                                action_mask[env_ix].cpu().numpy(  # 11
                                ) if action_mask is not None else None,
                                infos[env_ix]                     # 12
                            )
                        )

                    # mark this episode to terminated if done occurs once
                    if dones[env_ix]:
                        ep_terminated[env_ix] = True

                episode_starts = np.zeros((self.env.num_envs,), dtype=bool)
                current_obs = new_obs

            # now we flush all episode wise info to the 1-D buffer
            rollout_info = self._add_to_buffer(
                rollout_buffer, episode_wise_transitions, rollout_info)
            return rollout_info

        def _add_to_buffer(self, rollout_buffer, episode_wise_transitions, rollout_info):
            # if the reward function is batchable, we override the rewards here
            if isinstance(self.reward_fn, BatchedRewardFunction):
                compute_batched_rewards(
                    episode_wise_transitions, self.reward_fn)

            advantages_computed = False
            for ep_ix, transitions in enumerate(episode_wise_transitions):
                ep_length = len(transitions)
                total_reward = 0.0
                total_kl_reward = 0.0
                for transition_ix, (obs, action, task_reward, reward, kl_div, ep_start, value, log_prob, done, ref_log_prob, kl_reward, action_mask, info) in enumerate(transitions):
                    total_reward += task_reward
                    total_kl_reward += kl_reward
                    rollout_info["rollout_info/kl_div_mean"].append(kl_div)
                    rollout_info["rollout_info/log_prob"].append(log_prob)
                    rollout_info["rollout_info/ref_log_prob"].append(
                        ref_log_prob)
                    rollout_info["rollout_info/values"].append(value.numpy())

                    if not rollout_buffer.full:
                        rollout_buffer.add(obs, action, reward,
                                           ep_start, value, log_prob,
                                           action_masks=action_mask)

                    # if the buffer is full, compute advantages
                    if rollout_buffer.full and not advantages_computed:

                        # normalize the rewards
                        if self._norm_reward:
                            mean = rollout_buffer.rewards.mean()
                            std = rollout_buffer.rewards.std()
                            rollout_buffer.rewards = (
                                rollout_buffer.rewards - mean) / (std + 1e-8)

                        # we fetch the last value for the last time step
                        # values come from the next transitions's values
                        next_values = transitions[transition_ix +
                                                  1][6] if (transition_ix + 1) < ep_length else torch.tensor([0.0])

                        rollout_buffer.compute_returns_and_advantage(
                            last_values=next_values, dones=done)
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
            max_steps = env.unwrapped.get_attr(
                "max_steps", [0])[0]

            # get tokenizer
            tokenizer = env.unwrapped.get_attr("tokenizer", [0])
            tokenizer = tokenizer[0]

            # Switch to eval mode
            self.policy.set_training_mode(False)

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
                "rollout_info/values": []
            }
            while not rollout_buffer.full:
                # generate batch of rollouts
                rollout_info = self.generate_batch(rollout_buffer, tokenizer,
                                                   max_steps, rollout_info)

            # aggregate rollout info
            aggregated_rollout_info = {}
            for key, values in rollout_info.items():
                aggregated_rollout_info[key] = np.mean(values).item()
                aggregated_rollout_info[f"{key}_std"] = np.std(values).item()
            aggregated_rollout_info["rollout_info/kl_coeff"] = self._kl_controller.kl_coeff

            if self.tracker is not None:
                self.tracker.log_rollout_infos(aggregated_rollout_info)

            # adapt the KL coeff
            self._kl_controller.step(torch.tensor(
                aggregated_rollout_info["rollout_info/kl_div_mean"]))

            # sanity check 3: now, loop over the buffer
            # and check the log_probs and values match
            # for rollout_data in self.rollout_buffer.get(self.batch_size):
            #     actions = rollout_data.actions.long().flatten()
            #     values, log_prob, entropy = self.policy.evaluate_actions(
            #         rollout_data.observations, actions)

            #     assert torch.allclose(
            #         values.flatten(), rollout_data.old_values.flatten(), atol=1e-4)
            #     assert torch.allclose(
            #         log_prob, rollout_data.old_log_prob, atol=1e-4)
            return True

    # instantiate the wrapped alg
    alg = OnPolicyAlgText(alg_kwargs, kl_coeff, tracker,
                          target_kl, norm_reward)
    return alg