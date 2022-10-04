import copy
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import functional as F

from rl4lms.algorithms.common.algo_utils import conjugate_gradient_solver, flat_grad
from rl4lms.algorithms.trpo.policies import *
from rl4lms.envs.text_generation.logging_utils import Tracker


class TRPO(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization (TRPO)

    Paper: https://arxiv.org/abs/1502.05477
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    and Stable Baselines (TRPO from https://github.com/hill-a/stable-baselines)

    Introduction to TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size for the value function
    :param gamma: Discount factor
    :param cg_max_steps: maximum number of steps in the Conjugate Gradient algorithm
        for computing the Hessian vector product
    :param cg_damping: damping in the Hessian vector product computation
    :param line_search_shrinking_factor: step-size reduction factor for the line-search
        (i.e., ``theta_new = theta + alpha^i * step``)
    :param line_search_max_iter: maximum number of iteration
        for the backtracking line-search
    :param n_critic_updates: number of critic updates per policy update
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param target_kl: Target Kullback-Leibler divergence between updates.
        Should be small for stability. Values like 0.01, 0.05.
    :param sub_sampling_factor: Sub-sample the batch to make computation faster
        see p40-42 of John Schulman thesis http://joschu.net/docs/thesis.pdf
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
                    tracker: Tracker,
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cuda",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,  # entropy bonus is not used by TRPO
            vf_coef=0.0,  # value function is optimized separately
            max_grad_norm=0.0,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            if normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        # Conjugate gradients parameters
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        # Backtracking line search parameters
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        self.target_kl = target_kl
        self.n_critic_updates = n_critic_updates
        self.sub_sampling_factor = sub_sampling_factor

        if _init_setup_model:
            self._setup_model()

        self._tracker = tracker

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) -> Tuple[List[nn.Parameter], th.Tensor, th.Tensor, List[Tuple[int, ...]]]:
        """
        Compute actor gradients for kl div and surrogate objectives.

        :param kl_div: The KL divergence objective
        :param policy_objective: The surrogate objective ("classic" policy gradient)
        :return: List of actor params, gradients and gradients shape.
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # The policy objective is also called surrogate objective
        policy_objective_gradients = []
        # Contains the gradients of the KL divergence
        grad_kl = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters
        grad_shape = []
        # Contains the parameters which have non-zeros KL divergence gradients
        # The list is used during the line-search to apply the step to each parameters
        actor_params = []

        for name, param in self.policy.named_parameters():
            # Skip parameters related to value function based on name
            # this work for built-in policies only (not custom ones)
            if "value" in name:
                continue

            # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param.to(kl_div.device),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            # If the gradient is not zero (not None), we store the parameter in the actor_params list
            # and add the gradient and its shape to grad_kl and grad_shape respectively
            if kl_param_grad is not None:
                # If the parameter impacts the KL divergence (i.e. the policy)
                # we compute the gradient of the policy objective w.r.t to the parameter
                # this avoids computing the gradient if it's not going to be used in the conjugate gradient step
                policy_objective_grad, * \
                    _ = th.autograd.grad(
                        policy_objective.to(param.device), param, retain_graph=True, only_inputs=True)

                grad_shape.append(kl_param_grad.shape)
                grad_kl.append(kl_param_grad.reshape(-1))
                policy_objective_gradients.append(
                    policy_objective_grad.reshape(-1).to(kl_param_grad.device))
                actor_params.append(param)

        # Gradients are concatenated before the conjugate gradient step
        policy_objective_gradients = th.cat(policy_objective_gradients)
        grad_kl = th.cat(grad_kl)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        #self.policy.set_training_mode(True)
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        gather_device = self.policy.device

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,  # old values, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    None,  # returns, not used here
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                # batch_size is only used for the value function
                self.policy.reset_noise(actions.shape[0])

            # Note: is copy enough, no need for deepcopy?
            # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
            # directly to avoid PyTorch errors.
            with th.no_grad():
                old_distribution = copy.copy(self.policy.get_distribution(
                    rollout_data.observations, detach=True))

            distribution = self.policy.get_distribution(
                rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / \
                    (rollout_data.advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean()

            # KL divergence
            kl_div = kl_divergence(
                distribution.distribution, old_distribution.distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
                kl_div, policy_objective)

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(
                self.hessian_vector_product, actor_params, grad_kl)

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            # Maximal step length
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= th.matmul(
                search_direction, hessian_vector_product_fn(
                    search_direction, retain_graph=False)
            )
            line_search_max_step_size = th.sqrt(line_search_max_step_size)

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone().to(
                gather_device) for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):

                    start_idx = 0
                    # Applying the scaled step direction
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx: (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(
                        rollout_data.observations)
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()

                    # New KL-divergence
                    kl_div = kl_divergence(
                        distribution.distribution, old_distribution.distribution).mean()

                    # Constraint criteria:
                    # we need to improve the surrogate policy objective
                    # while being close enough (in term of kl div) to the old policy
                    if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(actor_params, original_actor_params):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(
                    rollout_data.observations)
                value_loss = F.mse_loss(
                    rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_objective",
                           np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success",
                           np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")

    def hessian_vector_product(
        self, params: List[nn.Parameter], grad_kl: th.Tensor, vector: th.Tensor, retain_graph: bool = True
    ) -> th.Tensor:
        """
        Computes the matrix-vector product with the Fisher information matrix.

        :param params: list of parameters used to compute the Hessian
        :param grad_kl: flattened gradient of the KL divergence between the old and new policy
        :param vector: vector to compute the dot product the hessian-vector dot product with
        :param retain_graph: if True, the graph will be kept after computing the Hessian
        :return: Hessian-vector dot product (with damping)
        """
        jacobian_vector_product = (grad_kl * vector).sum()
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TRPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OnPolicyAlgorithm:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
