import os
import copy
import hydra
import sys
import torch
import tqdm
from omegaconf import OmegaConf
from tensordict import TensorDict

from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import MarlGroupMapType

import torchrl.collectors



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
# from env.cache_guessing_game_env import CacheGuessingGameEnv
# from env.env_pettingzoo import CacheAttackerDefenderEnv
from env.macta_pettingzoo import CacheAttackerDetectorEnv
# from env.cache_attacker_detector_env import CacheAttackerDetectorEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from tensordict import TensorDictBase

from torchrl.envs import Compose, TransformedEnv, \
    EnvCreator, ParallelEnv, RewardSum, StepCounter, Transform, EnvBase
from torchrl.envs import set_exploration_type, ExplorationType
from torchrl.envs.utils import check_env_specs

import model_utils
from new_cache_ppo_transformer_model import CachePPOTransformerModel, CachePPOTransformerModel2

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import EGreedyModule
from torchrl.modules import (
    AdditiveGaussianWrapper,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
)
from tqdm import tqdm
from torchrl.record.loggers.wandb import WandbLogger
from pettingzoo.utils import agent_selector, wrappers


HERE = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path="../config",
    # config_name="new_macta"
    config_name="macta"
)
def main(cfg):

    
    print(f"workding_dir = {os.getcwd()}")

    # ========= Logger ========= #
    # logger = WandbLogger(
    #     exp_name="-".join(["macta", "exp0"]),
    #     config=cfg
    # )

    # ========= Save config ========= #
    # save the config
    # os.makedirs(f"{HERE}/saved_{logger.exp_name}", exist_ok=True)
    # torch.save(cfg, f"{HERE}/saved_{logger.exp_name}/cfg.pt")

    # ========= Extract config params (for efficiency) ========= #
    frames_per_batch = 200#262144
    total_frames = 1000
    # total_frames = cfg.collector.total_frames
    # num_epochs = cfg.num_epochs
    # eval_freq = cfg.eval_freq
    # device = cfg.device
    # env_config = cfg.env_config
    # env_config = OmegaConf.to_container(env_config)
    # num_workers = cfg.collector.num_workers
    # envs_per_collector = cfg.collector.envs_per_collector
    # preemptive_threshold = cfg.collector.preemptive_threshold
    # collector_device = cfg.collector.device
    # clip_grad_norm = cfg.loss.clip_grad_norm
    # save_freq = cfg.logger.save_frequency
    prefetch = cfg.prefetch
    batch_size = cfg.batch_size
    # replay_buffer_size = cfg.replay_buffer_size

    # ========= Env factory ========= #
    # We don't want to serialize the env, a constructor is sufficient.
    def make_env():
        # env = tictactoe_v3.env()
        env = CacheAttackerDetectorEnv(cfg.env_config)
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        # env = wrappers.OrderEnforcingWrapper(env)
        env = PettingZooWrapper(
            # env=CacheAttackerDetectorEnv(cfg.env_config),
            env=env,
            use_mask=True, # Must use it since one player plays at a time
            group_map=None, # # Use default for AEC (one group per player)
            )
        return env 

        # return TransformedEnv(
        #     env#, 
        #     # RewardSum(in_keys=[env.reward_key[1]], out_keys=[("opponent", "reward")]),
        #     # StepCounter(),
        # )

    # def test_make_env():
    #     return TransformedEnv(
    #         GymWrapper(CacheGuessingGameEnv(cfg.env_config), device=cfg.train_device),
    #         Compose(
    #             RewardSum(),
    #             StepCounter(),
    #         )
    #     )
    
    env = make_env()
    # print("reward_keys:", type(env.reward_keys))
    # print(env.reset()["episode_reward"])
    # print(env.group_map)
    # print(env.rollout(5))
    # print("observation_spec:", env.observation_spec)
    # print("acion_spec:", env.action_spec)
    dummy_env = make_env()
    dummy_env_d = make_env()
    env = TransformedEnv(env, RewardSum(
        in_keys=env.reward_keys,
        reset_keys=["_reset"] * len(env.group_map.keys()),
        ),
    )

    check_env_specs(env)
    print("++++++++++++++env made")

    policy_modules = {}
    for group, agents in env.group_map.items():
        if 'opponent' in agents:    
            cfg.model_config["output_dim"] = 16
        else:
            cfg.model_config["output_dim"] = 2
        # print("inside the for loop: ", group, agents)
        train_model = model_utils.get_model(
            cfg.model_config, cfg.env_config.window_size,
            cfg.model_config["output_dim"]
        ).to(device)
        policy_net = train_model.get_actor()

        # policy_net = CachePPOTransformerModel(**cfg.model_config).to(cfg.train_device)

        # Wrap the neural network in a :class:`~tensordict.nn.TensorDictModule`.
        # This is simply a module that will read the ``in_keys`` from a tensordict, feed them to the
        # neural networks, and write the
        # outputs in-place at the ``out_keys``.

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "logits")],
        )  # We just name the input and output that the network will read and write to the input tensordict
        policy_modules[group] = policy_module.to(cfg.train_device)


    policies = {}
    for group, _agents in env.group_map.items():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class= torch.distributions.Categorical,
            distribution_kwargs={},
            return_log_prob=False,
            # log_prob_key=(group, "sample_log_prob"),
        )
        policies[group] = policy.to(cfg.train_device)

    exploration_policies = {}
    for group, _agents in env.group_map.items():
        # exploration_policy = AdditiveGaussianWrapper(
        #     policies[group],
        #     sigma_init=0.9,
        #     sigma_end=0.1,
        #     annealing_num_steps=total_frames // 2,
        #     action_key=(group, "action"),
        # )
        # exploration_policies[group] = exploration_policy.to(cfg.train_device)
        exploration_policies[group] = policies[group].to(cfg.train_device)


    critics = {}
    for group, agents in env.group_map.items():
        # cat_module = TensorDictModule(
        #     lambda obs, action: torch.cat([obs, action], dim=-1),
        #     in_keys=[(group, "observation"), (group, "action")],
        #     out_keys=[(group, "obs_action")],
        # )
        
        cfg.model_config["output_dim"] = 1
        # print("inside the for loop: ", group, agents)

        critic_net = CachePPOTransformerModel2(**cfg.model_config).to(cfg.train_device)

        critic_module = TensorDictModule(
            module=critic_net,
            in_keys=[(group, "observation")],  # Read ``(group, "obs_action")``
            out_keys=[
                (group, "state_action_value")
            ],  # Write ``(group, "state_action_value")``
        )

        critics[group] = critic_module.to(cfg.train_device)#TensorDictSequential(
        #     cat_module, critic_module
        # ).to(cfg.train_device)


    # for group, _agents in env.group_map.items():
    #     print("Running policy:", policies[group](env.reset().to(cfg.train_device)))
    #     print("Running value:", critics[group](env.reset().to(cfg.train_device)))

    agents_exploration_policy = TensorDictSequential(*exploration_policies.values()).to(cfg.train_device)
    
    
    collector = torchrl.collectors.SyncDataCollector(
        env,
        agents_exploration_policy,
        # policies,
        device=cfg.train_device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    replay_buffers = {}
    for group, _agents in env.group_map.items():
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(262144, device=cfg.train_device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
            prefetch=prefetch,
        )
        replay_buffers[group] = replay_buffer

    losses = {}
    for group, _agents in env.group_map.items():
        loss_module = DDPGLoss(
            actor_network=policies[group],  # Use the non-explorative policies
            value_network=critics[group],
            delay_value=True,  # Whether to use a target network for the value
            loss_function="l2",
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=0.99)

        losses[group] = loss_module

    target_updaters = {
        group: SoftUpdate(loss, tau=0.005) for group, loss in losses.items()
    }

    optimisers = {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(), lr=1e-4
            ),
            "loss_value": torch.optim.Adam(
                loss.value_network_params.flatten_keys().values(), lr=1e-4
            ),
        }
        for group, loss in losses.items()
    }


    def process_batch(batch: TensorDictBase) -> TensorDictBase:
        """
        If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
        `"terminated"` and `"done"`.
        This is needed to present them with the same shape as the reward to the loss.
        """
        for group in env.group_map.keys():
            keys = list(batch.keys(True, True))
            group_shape = batch.get_item_shape(group)
            nested_done_key = ("next", group, "done")
            nested_terminated_key = ("next", group, "terminated")
            if nested_done_key not in keys:
                batch.set(
                    nested_done_key,
                    batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
                )
            if nested_terminated_key not in keys:
                batch.set(
                    nested_terminated_key,
                    batch.get(("next", "terminated"))
                    .unsqueeze(-1)
                    .expand((*group_shape, 1)),
                )
        return batch

    pbar = tqdm(
        total=10,
        desc=", ".join(
            [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
        ),
    )
    print("what is happening? 1")
    episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
    print("what is happening? 2")
    train_group_map = copy.deepcopy(env.group_map)
    print("what is happening? 3")

    # Training/collection iterations
    for iteration, batch in enumerate(collector):
        print("is it progressing? -1")
        current_frames = batch.numel()
        print("is it progressing? 0")
        batch = process_batch(batch)  # Util to expand done keys if needed
        print("is it progressing? 1")
        # Loop over groups
        for group in train_group_map.keys():
            print("is it progressing? 2")
            group_batch = batch.exclude(
                *[
                    key
                    for _group in env.group_map.keys()
                    if _group != group
                    for key in [_group, ("next", _group)]
                ]
            )  # Exclude data from other groups
            group_batch = group_batch.reshape(
                -1
            )  # This just affects the leading dimensions in batch_size of the tensordict
            replay_buffers[group].extend(group_batch)

            for _ in range(100):
                subdata = replay_buffers[group].sample()
                loss_vals = losses[group](subdata)

                for loss_name in ["loss_actor", "loss_value"]:
                    loss = loss_vals[loss_name]
                    optimiser = optimisers[group][loss_name]

                    loss.backward()

                    # Optional
                    params = optimiser.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

                    optimiser.step()
                    optimiser.zero_grad()

                # Soft-update the target network
                target_updaters[group].step()

            # Exploration sigma anneal update
            exploration_policies[group].step(current_frames)

        # Stop training a certain group when a condition is met (e.g., number of training iterations)
        if iteration == iteration_when_stop_training_evaders:
            del train_group_map["agent"]

        # Logging
        for group in env.group_map.keys():
            episode_reward_mean = (
                batch.get(("next", group, "episode_reward"))[
                    batch.get(("next", group, "done"))
                ]
                .mean()
                .item()
            )
            episode_reward_mean_map[group].append(episode_reward_mean)

        pbar.set_description(
            ", ".join(
                [
                    f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                    for group in env.group_map.keys()
                ]
            ),
            refresh=False,
        )
        pbar.update()


  

if __name__ == "__main__":
    main()
