import os
import sys
import copy
import logging
import time
import subprocess
import platform
import psutil
import random
import tqdm

import hydra
import threading
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Dict
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict
import rlmeta.utils.nested_utils as nested_utils
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn 
import torch.multiprocessing as mp
import re
import numpy as np
# import tqdm
# from omegaconf import OmegaConf
# from tensordict import TensorDict
# import torchrl.collectors

import rlmeta.envs.gym_wrappers as gym_wrappers
import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.remote_utils as remote_utils

from rlmeta.agents.agent import AgentFactory
from rlmeta.core.replay_buffer import ReplayBuffer, make_remote_replay_buffer
from rlmeta.core.server import Server, ServerList
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer_model_pool import CachePPOTransformerModelPool, wrap_downstream_model
from model.transformer_model_pool_defender import CachePPOTransformerModelPoolDefender, wrap_downstream_model
from model.transformer_model_pool_student import CachePPOTransformerModelPoolStudent, wrap_downstream_model
from env.cache_attacker_defender_env_factory import CacheAttackerDefenderEnvFactory

from utils.ma_metric_callbacks import MACallbacks
from utils.wandb_logger import WandbLogger, stats_filter
from utils.controller import Phase, Controller
from utils.maloop import LoopList, MAParallelLoop

from utils.trace_parser import load_trace

from agent import RandomAgent, BenignAgent, SpecAgent, PPOAgent
from agent import SpecAgentFactory

@dataclass
class ReplayBufferEntry:
    state: torch.Tensor
    action: torch.Tensor

import random

class CustomReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # Track the position to overwrite when at capacity

    def append(self, entry: ReplayBufferEntry):
        """Add a new entry to the buffer, replacing the oldest if full."""
        if len(self.buffer) < self.capacity:
            # If there's space, add the entry
            self.buffer.append(entry)
        else:
            # If at capacity, overwrite the oldest entry
            self.buffer[self.position] = entry
        # Update position to point to the next entry to be overwritten
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of entries from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


def evaluate_agent(env_eval, eval_agents, metrics_multidataset):
    metrics = run_loops(env_eval, eval_agents)
    metrics_mean = metrics.dict()
    for k in metrics_mean.keys(): 
        metrics_mean[k] = metrics_mean[k]['mean']
    metrics_multidataset.extend(metrics_mean)
    return metrics_multidataset, metrics

def evaluate_student(env_eval, eval_agents, metrics_multidataset):
    metrics = run_loops_student(env_eval, eval_agents)
    metrics_mean = metrics.dict()
    for k in metrics_mean.keys(): 
        metrics_mean[k] = metrics_mean[k]['mean']
    metrics_multidataset.extend(metrics_mean)
    return metrics_multidataset, metrics


def buffer_update_thread(rb_d, spec_agents_list, attacker_agent, defender_agent, env):
    """Function to continuously update the replay buffer in the background."""
    while True:
        try:
            agents = {"attacker": attacker_agent, "defender": defender_agent, "benign": random.choice(spec_agents_list)}
            reset_data, reset_benign_index_counter = agents['benign'].get_reset_data(env.env.reset_mode)
            if reset_benign_index_counter > 1:
                reset_data = None

            timestep = env.reset(reset_data=reset_data)

            if env.env.opponent_agent == 'attacker':
                if agents['benign'].reset_benign_index_counter == 1:
                    agents['benign'].reset_benign_index_counter = agents['benign'].reset_benign_index_freq
                else:
                    agents['benign'].reset_benign_index_counter -= 1
                agents['benign'].step -= 64
            #act from def
            # print("check the data from spec agent: ", agents['benign'].reset_benign_index_counter, agents['benign'].reset_benign_index_freq, env.env.opponent_agent)
            if env.env.opponent_agent == 'benign' and env.env.reset_mode == 'env' and agents['benign'].reset_benign_index_counter == 1:
                warmup_acion = {}
                observs = timestep['defender'].observation
                observs = observs.unsqueeze(0)
                # print('this is the type i am looking for ', type(observs), env.env.opponent_agent)
                for i in range(0, env.env.reset_warmup_length):
                    warmup_acion['benign'], info, idx = agents['benign'].get_warmup_data()
                    # print("data from benign agent: ", warmup_acion['benign'], info, idx)
                    warmup_acion['defender'] = agents['defender'].fake_act(observs)[0]
                    warmup_acion['attacker'] = 0
                    observs = env.env.fake(warmup_acion, info)
                    # print("defender's observations: ", observs[0])
                    observs = torch.tensor(observs)
                    observs = observs.unsqueeze(0)
                    # print("look here", observs)
                    # rb_d.append({"state": observs})

            # print("*******************************",rb_d.size)
            # print("*******************************")
            # print("******************************* filling the replay buffer")
            
            
            for agent_name, agent in agents.items():
                agent.observe_init(timestep[agent_name])
            while not timestep["__all__"].done:
                # Model server requires a batch_dim, so unsqueeze here for local runs.
                actions = {}
                for agent_name, agent in agents.items():
                    timestep[agent_name].observation.unsqueeze_(0)
                    # print(agent_name, agent)
                    # print(timestep[agent_name])
                    action = agent.act(timestep[agent_name])
                    # print(action)
                    actions.update({agent_name:action})
                timestep = env.step(actions)
                bin_rep = format(int(actions["defender"].action), '016b') 
                detector_action = torch.tensor([int(bit) for bit in bin_rep], dtype=torch.float32)
                entry = ReplayBufferEntry(state=timestep['defender'].observation.squeeze(), action=detector_action)
                rb_d.append(entry)

                # print("here:  ", timestep['defender'].observation)
                # print("and then: ", actions['defender'].info['logpi'], actions['defender'].info['v'])
                for agent_name, agent in agents.items():
                    agent.observe(actions[agent_name], timestep[agent_name])
        except Exception as e:
            print("Exception in buffer update thread:")
            traceback.print_exc()  # Print the full traceback for debugging
            break


def prepare_model_pool(attacker_model, cfg):
    directory = cfg.prepare_pool
    attacker_pattern = re.compile(r'^attacker-(\d+)\.pth$')
    
    try:
        for filename in os.listdir(directory):
            attacker_match = attacker_pattern.match(filename)
            
            if attacker_match:
                number = int(attacker_match.group(1))
                if number % 10 == 9:
                    full_path = os.path.join(directory, filename)
                    attacker_add_pool = torch.load(full_path, map_location=cfg.train_device)
                    attacker_model.push_to_history(attacker_add_pool)
    
    except FileNotFoundError:
        print(f"Directory {directory} not found.")
    return attacker_model

def get_git_commit_id():
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_id
    except Exception as e:
        logging.error(f"Error retrieving Git commit ID: {e}")
        return "Unknown"
def get_hardware_info():
    try:
        info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'memory': psutil.virtual_memory().total
        }
        return info
    except Exception as e:
        logging.error(f"Error retrieving hardware info: {e}")
        return {}

def parse_range_file(range_file):
    range_dict = {}
    with open(range_file, 'r') as infile:
        for line in infile:
            filename_part, range_part = line.split(": lines ")
            filename = filename_part.strip()
            mod_number = re.search(r'_Mod_(\d+)', filename)
            if mod_number:
                mod_key = mod_number.group(1)  # Get the number after _Mod
            else:
                continue  # If no _Mod number is found, skip the line
            start_line, end_line = range_part.split("–")
            start_line = int(start_line.strip())
            end_line = int(end_line.strip())
            if end_line - start_line < 500:
                continue
            offset = 70 * 4#(self.env_config.get('reset_benign_index_freq', 0)+1)
            range_dict[mod_key] = (start_line-1, end_line-offset)
    return range_dict


def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)

def run_loop(env, agents, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    detector_count = 0.0
    detector_acc = 0.0
    num_total_guess = 0.0
    num_total_correct_guess = 0.0
    defender_episode_return = 0.0
    benign_hit_rate = 0.0
    locked_lines = 0.0
    hit_rate = 0.0
    memory_access_count = 0  

    reset_data, reset_benign_index_counter = agents['benign'].get_reset_data(env.env.reset_mode)
    if reset_benign_index_counter > 1:
        reset_data = None
    if victim_addr == -1:
        timestep = env.reset(reset_data=reset_data)
    else:
        timestep = env.reset(victim_address=victim_addr, reset_data=reset_data)

    if agents['benign'].reset_benign_index_counter == 1:
        agents['benign'].reset_benign_index_counter = agents['benign'].reset_benign_index_freq
    else:
        agents['benign'].reset_benign_index_counter -= 1
    agents['benign'].step -= 64
    #act from def
    # print("check the data from spec agent: ", agents['benign'].reset_benign_index_counter, agents['benign'].reset_benign_index_freq)
    
    if env.env.opponent_agent == 'benign' and env.env.reset_mode == 'env' and agents['benign'].reset_benign_index_counter == 1:
        warmup_acion = {}
        observs = timestep['defender'].observation
        observs = observs.unsqueeze(0)
        # print('this is the type i am looking for ', type(observs), env.env.opponent_agent)
        for i in range(0, env.env.reset_warmup_length):
            warmup_acion['benign'], info, idx = agents['benign'].get_warmup_data()
            # print("data from benign agent: ", warmup_acion['benign'], info, idx, d)
            warmup_acion['defender'] = agents['defender'].fake_act(observs)[0]
            warmup_acion['attacker'] = 0
            observs = env.env.fake(warmup_acion, info)
            # print("defender's observations: ", observs[0])
            observs = torch.tensor(observs)
            observs = observs.unsqueeze(0)

    for agent_name, agent in agents.items():
        agent.observe_init(timestep[agent_name])
    while not timestep["__all__"].done:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        actions = {}
        for agent_name, agent in agents.items():
            timestep[agent_name].observation.unsqueeze_(0)
            action = agent.act(timestep[agent_name])
            # Unbatch the action.
            if isinstance(action, tuple):
                action = Action(action[0], action[1])
            if not isinstance(action.action, (int, np.int64)):
                action = unbatch_action(action)
            actions.update({agent_name:action})
        timestep = env.step(actions)

        for agent_name, agent in agents.items():
            agent.observe(actions[agent_name], timestep[agent_name])
        
        episode_length += 1
        episode_return += timestep['attacker'].reward
        is_guess = timestep['attacker'].info.get("is_guess",0)
        correct_guess = timestep['attacker'].info.get("guess_correct",0)
        num_total_guess += is_guess
        num_total_correct_guess += correct_guess

        if timestep['benign'].info["opponent"] == "benign":
            if timestep['benign'].info["hit_miss"] == "miss" or timestep['benign'].info["hit_miss"] == "hit" : 
                memory_access_count += 1
        locked_lines += timestep['benign'].info["locked_count"]
        if (timestep['benign'].info["hit_miss"] == "hit") and (timestep['benign'].info["opponent"] == "benign"):
            hit_rate += 1


        defender_episode_return += timestep['defender'].reward
        if num_total_guess != 0:
            ep_ret = episode_length / num_total_guess
        else:
            ep_ret = episode_length

        try:
            detector_action = actions['defender'].action.item()
        except:
            detector_action = actions['defender'].action
        if timestep["__all__"].done and detector_action ==1:
            detector_count += 1
            #print(timestep["detector"])
        detector_accuracy = detector_count
        if memory_access_count == 0: 
            h_rate = 0 
        else: 
            h_rate = float(hit_rate/memory_access_count)


    metrics = {
        "episode_length": ep_ret,
        # "episode_length": episode_l ength,
        "locked_lines": float(locked_lines/episode_length),
        "hit_rate": h_rate,
        "episode_return": episode_return,
        "defender_episode_return": defender_episode_return,
        "num_total_guess": num_total_guess,
        "num_total_correct_guess": num_total_correct_guess,
        "attacker_correct_rate": float(num_total_correct_guess / num_total_guess) if num_total_guess != 0 else 0
    }
    return metrics

def run_loop_student(env, agents, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    detector_count = 0.0
    detector_acc = 0.0
    num_total_guess = 0.0
    num_total_correct_guess = 0.0
    defender_episode_return = 0.0
    benign_hit_rate = 0.0
    locked_lines = 0.0
    hit_rate = 0.0
    memory_access_count = 0  
    # print("inside run loop student. type: ", type(agents), type(agents['defender']))

    reset_data, reset_benign_index_counter = agents['benign'].get_reset_data(env.env.reset_mode)
    if reset_benign_index_counter > 1:
        reset_data = None
    if victim_addr == -1:
        timestep = env.reset(reset_data=reset_data)
    else:
        timestep = env.reset(victim_address=victim_addr, reset_data=reset_data)

    if agents['benign'].reset_benign_index_counter == 1:
        agents['benign'].reset_benign_index_counter = agents['benign'].reset_benign_index_freq
    else:
        agents['benign'].reset_benign_index_counter -= 1
    agents['benign'].step -= 64

    if env.env.opponent_agent == 'benign' and env.env.reset_mode == 'env' and agents['benign'].reset_benign_index_counter == 1:
        warmup_acion = {}
        observs = timestep['defender'].observation
        observs = observs.unsqueeze(0)
        # print('this is the type i am looking for ', type(observs), env.env.opponent_agent)
        for i in range(0, env.env.reset_warmup_length):
            warmup_acion['benign'], info, idx = agents['benign'].get_warmup_data()
            # print("data from benign agent: ", warmup_acion['benign'], info, idx, d)
            gathered_actions = []
            for student in agents['defender']:
                gathered_actions.append(student.fake_act(observs)[0])
            binary_string = ''.join(str(bit) for bit in gathered_actions)
            decimal_value = int(binary_string, 2)
            warmup_acion['defender'] = decimal_value#agent['defender'].fake_act(observs)[0]
            warmup_acion['attacker'] = 0
            observs = env.env.fake(warmup_acion, info)
            # print("defender's observations: ", observs[0])
            observs = torch.tensor(observs)
            observs = observs.unsqueeze(0)
    
    # for agent_name, agent in agents.items():
    #     agent.observe_init(timestep[agent_name])
    for agent_name, agent in agents.items():
        if agent_name != 'defender':
            agent.observe_init(timestep[agent_name])

    while not timestep["__all__"].done:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        actions = {}
        for agent_name, agent in agents.items():
            timestep[agent_name].observation.unsqueeze_(0)
            if agent_name == 'defender':
                gathered_actions = []
                for agnt in agent:
                    gathered_actions.append(int(agnt.act(timestep[agent_name]).action))
                binary_string = ''.join(str(bit) for bit in gathered_actions)
                decimal_value = int(binary_string, 2)
                # print(agent_name, agent, gathered_actions, decimal_value)
                action = Action(decimal_value, info={"logpi": 1, "v": 2})
            else:
                action = agent.act(timestep[agent_name])
            # Unbatch the action.
            if isinstance(action, tuple):
                action = Action(action[0], action[1])
            if not isinstance(action.action, (int, np.int64)):
                action = unbatch_action(action)
            actions.update({agent_name:action})
        timestep = env.step(actions)

        for agent_name, agent in agents.items():
            if agent_name != 'defender':
                agent.observe(actions[agent_name], timestep[agent_name])
        
        episode_length += 1
        episode_return += timestep['attacker'].reward
        is_guess = timestep['attacker'].info.get("is_guess",0)
        correct_guess = timestep['attacker'].info.get("guess_correct",0)
        num_total_guess += is_guess
        num_total_correct_guess += correct_guess

        if timestep['benign'].info["opponent"] == "benign":
            if timestep['benign'].info["hit_miss"] == "miss" or timestep['benign'].info["hit_miss"] == "hit" : 
                memory_access_count += 1
        locked_lines += timestep['benign'].info["locked_count"]
        if (timestep['benign'].info["hit_miss"] == "hit") and (timestep['benign'].info["opponent"] == "benign"):
            hit_rate += 1


        defender_episode_return += timestep['defender'].reward
        if num_total_guess != 0:
            ep_ret = episode_length / num_total_guess
        else:
            ep_ret = episode_length

        try:
            detector_action = actions['defender'].action.item()
        except:
            detector_action = actions['defender'].action
        if timestep["__all__"].done and detector_action ==1:
            detector_count += 1
            #print(timestep["detector"])
        detector_accuracy = detector_count
        if memory_access_count == 0: 
            h_rate = 0 
        else: 
            h_rate = float(hit_rate/memory_access_count)


    metrics = {
        "episode_length": ep_ret,
        # "episode_length": episode_l ength,
        "locked_lines": float(locked_lines/episode_length),
        "hit_rate": h_rate,
        "episode_return": episode_return,
        "defender_episode_return": defender_episode_return,
        "num_total_guess": num_total_guess,
        "num_total_correct_guess": num_total_correct_guess,
        "attacker_correct_rate": float(num_total_correct_guess / num_total_guess) if num_total_guess != 0 else 0
    }
    return metrics

def run_loops(env: Env,
              agent: PPOAgent,
              ) -> StatsDict:
    # env.seed(seed)
    metrics = StatsDict()
    for i in tqdm.tqdm(range(50)):
        cur_metrics = run_loop(env, agent, victim_addr=-1)
        metrics.extend(cur_metrics)

    return metrics

def run_loops_student(env: Env,
              agent: PPOAgent,
              ) -> StatsDict:
    # env.seed(seed)
    metrics = StatsDict()
    for i in tqdm.tqdm(range(50)):
        cur_metrics = run_loop_student(env, agent, victim_addr=-1)
        metrics.extend(cur_metrics)

    return metrics

def float_weight_init(m):
    if isinstance(m, nn.Linear):
        device = m.weight.device
        torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)  # Initialize biases to 0


def train_single_student(student_agent, rb_d, teacher_bit_index, window_size_s, training_steps=1, device="cuda:0"):
    acc_loss = 0
    grad_value = 0
    isvm_loss = 0
    l2_loss = 0
    ent_loss = 0
    for training_step in range(training_steps):
        # Sample batch from the replay buffer
        batch = rb_d.sample(batch_size=64)
        states = torch.stack([entry.state[..., :] for entry in batch]).to(device)
        # cleaned_states = torch.stack([
        #     entry.state[~torch.all(entry.state == 0, dim=1)] for entry in batch
        # ]).to(device)
        # cleaned_states = cleaned_states[..., :-1]
        cleaned_state = torch.stack([entry.state[:window_size_s, :] for entry in batch]).to(device)
        # print(cleaned_state)
        # cleaned_state = state[:4]
        # print('&&&&&&&&&&&&&&&&')
        # print(states)
        # print(cleaned_state)
        # print('&&&&&&&&&&&&&&&&')
        teacher_actions = torch.stack([entry.action for entry in batch]).to(device)
        teacher_bits = [teacher_actions[:, i] for i in range(16)]
        # print("is batch size an issue?")
        # Act and train
        # a = student_agent.student_act(states)
        # acc_loss += student_agent.student_train(a.squeeze(), teacher_bits[teacher_bit_index])
        l, g, isvm_l, l2_l, ent_l = student_agent.student_train(cleaned_state, teacher_bits[teacher_bit_index])
        acc_loss += l
        grad_value += g
        isvm_loss += isvm_l
        l2_loss += l2_l
        ent_loss += ent_l
        # print("is it working?", l, g)
    return acc_loss/training_steps, grad_value/training_steps, isvm_loss/training_steps, l2_loss/training_steps, ent_loss/training_steps
        



@hydra.main(config_path="../config", config_name="macta")
def main(cfg):
    wandb_logger = WandbLogger(project="macta", config=cfg)
    print(f"workding_dir = {os.getcwd()}")
    my_callbacks = MACallbacks()
    logging.basicConfig(level=logging.INFO)
    logging.info(hydra_utils.config_to_json(cfg))
    git_commit_id = get_git_commit_id()
    logging.info(f"Git Commit ID: {git_commit_id}")

    # Log hardware information
    hardware_info = get_hardware_info()
    for key, value in hardware_info.items():
        logging.info(f"{key}: {value}")
   


    # =========================================================================
    # 50% benign, 50% attacker, for detector training
    env_fac = CacheAttackerDefenderEnvFactory(cfg.env_config)
    # 0% benign, 100% attacker, for attacker training and evaluation
    unbalanced_env_config = copy.deepcopy(cfg.env_config)
    unbalanced_env_config["opponent_weights"] = [0,1]
    env_fac_unbalanced = CacheAttackerDefenderEnvFactory(unbalanced_env_config)
    # 100% benign, 0% attacker, for detector evaluation (false positive)
    benign_env_config = copy.deepcopy(cfg.env_config)
    benign_env_config["opponent_weights"] = [1,0]
    env_fac_benign = CacheAttackerDefenderEnvFactory(benign_env_config)
    # =========================================================================

    #### Define model
    # =========================================================================
    # env = env_fac(0)
    env = env_fac_unbalanced(0)
    env_eval = env_fac_unbalanced(0)

    #### attacker model
    cfg.model_config["output_dim"] = env.action_space.n
    attacker_model = CachePPOTransformerModelPool(**cfg.model_config).to(cfg.train_device)
    attacker_params = torch.load(cfg.attacker_checkpoint, map_location=cfg.train_device)
    attacker_model.load_state_dict(attacker_params)
    attacker_model.eval()

    attacker_model = prepare_model_pool(attacker_model, cfg)
    infer_attacker = copy.deepcopy(attacker_model).to(cfg.infer_device)
    infer_attacker.eval()

    attacker_agent = PPOAgent(attacker_model, deterministic_policy=True)
    attacker_agent_infer = PPOAgent(infer_attacker, deterministic_policy=True)


    #### defender model
    cfg.model_config["step_dim"] += 2
    defender_model = CachePPOTransformerModelPoolDefender(**cfg.model_config).to(cfg.train_device)
    d_params = torch.load(cfg.defender_checkpoint, map_location=cfg.train_device)
    defender_model.load_state_dict(d_params)
    defender_model.eval()
    infer_defender = copy.deepcopy(defender_model).to(cfg.infer_device_d)
    infer_defender.eval()
    


    ### spec agents
    spec_agents_list = []
    for trace_file in cfg.trace_files:
        root, ext = os.path.splitext(trace_file)
        range_file = root + '_ranges' + ext
        range_info_dict = parse_range_file(range_file)
        spec_trace, cache_data = load_trace(trace_file, limit=cfg.trace_limit, legacy_trace_format=cfg.legacy_trace_format)
        agent = SpecAgent(env_config=cfg.env_config, range_info_dict=range_info_dict, trace=spec_trace, cache_data=cache_data, legacy_trace_format=cfg.legacy_trace_format)
        spec_agents_list.append(agent)
        print("agent added for ", trace_file)

    ### student models
    student_model1 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model1.apply(float_weight_init)
    student_model2 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model2.apply(float_weight_init)
    student_model3 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model3.apply(float_weight_init)
    student_model4 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model4.apply(float_weight_init)
    student_model5 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model5.apply(float_weight_init)
    student_model6 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model6.apply(float_weight_init)
    student_model7 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model7.apply(float_weight_init)
    student_model8 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model8.apply(float_weight_init)

    student_model9 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model9.apply(float_weight_init)
    student_model10 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model10.apply(float_weight_init)
    student_model11 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model11.apply(float_weight_init)
    student_model12 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model12.apply(float_weight_init)
    student_model13 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model13.apply(float_weight_init)
    student_model14 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model14.apply(float_weight_init)
    student_model15 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model15.apply(float_weight_init)
    student_model16 = CachePPOTransformerModelPoolStudent(**cfg.model_config).to(cfg.train_device)
    student_model16.apply(float_weight_init)

    
    print("done? ", len(spec_agents_list))

    ####### training preparation
    # make a replay buffer and fill it
    rb_d = CustomReplayBuffer(capacity=cfg.replay_buffer_size)#ReplayBuffer(capacity=cfg.replay_buffer_size)
    
    defender_agent = PPOAgent(defender_model, deterministic_policy=True)
    defender_agent_infer = PPOAgent(infer_defender, deterministic_policy=True)
    student_agent1 = PPOAgent(student_model1, deterministic_policy=True)
    student_agent2 = PPOAgent(student_model2, deterministic_policy=True)
    student_agent3 = PPOAgent(student_model3, deterministic_policy=True)
    student_agent4 = PPOAgent(student_model4, deterministic_policy=True)
    student_agent5 = PPOAgent(student_model5, deterministic_policy=True)
    student_agent6 = PPOAgent(student_model6, deterministic_policy=True)
    student_agent7 = PPOAgent(student_model7, deterministic_policy=True)
    student_agent8 = PPOAgent(student_model8, deterministic_policy=True)

    student_agent9 = PPOAgent(student_model9, deterministic_policy=True)
    student_agent10 = PPOAgent(student_model10, deterministic_policy=True)
    student_agent11 = PPOAgent(student_model11, deterministic_policy=True)
    student_agent12 = PPOAgent(student_model12, deterministic_policy=True)
    student_agent13 = PPOAgent(student_model13, deterministic_policy=True)
    student_agent14 = PPOAgent(student_model14, deterministic_policy=True)
    student_agent15 = PPOAgent(student_model15, deterministic_policy=True)
    student_agent16 = PPOAgent(student_model16, deterministic_policy=True)
    
    student_agents = [student_agent1, student_agent2, student_agent3, student_agent4, student_agent5, student_agent6, 
                    student_agent7, student_agent8, student_agent9, student_agent10, student_agent11, student_agent12,
                    student_agent13, student_agent14, student_agent15, student_agent16]

    # defender_agent.connect()
    #set history
    defender_agent.set_use_history(False)
    attacker_agent.set_use_history(True)
    agents = {"attacker": attacker_agent, "defender": defender_agent, "benign": random.choice(spec_agents_list)}
    for _ in range(1):
        reset_data, reset_benign_index_counter = agents['benign'].get_reset_data(env.env.reset_mode)
        # print("reset index is : ", reset_benign_index_counter)
        if reset_benign_index_counter > 1:
            reset_data = None

        timestep = env.reset(reset_data=reset_data)

        if env.env.opponent_agent == 'attacker':
            if agents['benign'].reset_benign_index_counter == 1:
                agents['benign'].reset_benign_index_counter = agents['benign'].reset_benign_index_freq
            else:
                agents['benign'].reset_benign_index_counter -= 1
            agents['benign'].step -= 64
        #act from def
        # print("check the data from spec agent: ", agents['benign'].reset_benign_index_counter, agents['benign'].reset_benign_index_freq, env.env.opponent_agent)
        if env.env.opponent_agent == 'benign' and env.env.reset_mode == 'env' and agents['benign'].reset_benign_index_counter == 1:
            warmup_acion = {}
            observs = timestep['defender'].observation
            observs = observs.unsqueeze(0)
            # print('this is the type i am looking for ', type(observs), env.env.opponent_agent)
            for i in range(0, env.env.reset_warmup_length):
                warmup_acion['benign'], info, idx = agents['benign'].get_warmup_data()
                # print("data from benign agent: ", warmup_acion['benign'], info, idx)
                warmup_acion['defender'] = agents['defender'].fake_act(observs)[0]
                warmup_acion['attacker'] = 0
                observs = env.env.fake(warmup_acion, info)
                # print("defender's observations: ", observs[0])
                observs = torch.tensor(observs)
                observs = observs.unsqueeze(0)
                # print("look here", observs)
                # rb_d.append({"state": observs})

        # print("*******************************",rb_d.size)
        # print("*******************************")
        # print("******************************* filling the replay buffer")
        
        
        for agent_name, agent in agents.items():
            agent.observe_init(timestep[agent_name])
        while not timestep["__all__"].done:
            # Model server requires a batch_dim, so unsqueeze here for local runs.
            actions = {}
            for agent_name, agent in agents.items():
                timestep[agent_name].observation.unsqueeze_(0)
                # print(agent_name, agent)
                # print(timestep[agent_name])
                action = agent.act(timestep[agent_name])
                # print(action)
                actions.update({agent_name:action})
            timestep = env.step(actions)
            bin_rep = format(int(actions["defender"].action), '016b') 
            detector_action = torch.tensor([int(bit) for bit in bin_rep], dtype=torch.float32)
            # print("detector action going in the replay buffer; ", detector_action, bin_rep, actions["defender"].action)
            entry = ReplayBufferEntry(state=timestep['defender'].observation.squeeze(), action=detector_action)
            rb_d.append(entry)

            # print("here:  ", timestep['defender'].observation)
            # print("and then: ", actions['defender'].info['logpi'], actions['defender'].info['v'])
            for agent_name, agent in agents.items():
                agent.observe(actions[agent_name], timestep[agent_name])


    print("!!!!!", rb_d.sample(batch_size=1))
    print("done2?", len(rb_d))
    #replay buffer filled. now we can start training and inserting new data to the replay buffer
    buffer_thread = threading.Thread(target=buffer_update_thread, args=(rb_d, spec_agents_list, attacker_agent, defender_agent, env))
    buffer_thread.daemon = True  # Daemonize thread to exit when main program exits
    buffer_thread.start()

    print("done3!")
    total_params = sum(p.numel() for p in student_agent8.model.parameters())
    print(f"Total number of parameters: {total_params}")
    time.sleep(60)
    #start actual training
    device = cfg.train_device
    # torch.save(student_agents[0].model.state_dict(), f"isvm1-00.pth")
    # torch.save(student_agents[1].model.state_dict(), f"isvm2-00.pth")
    # torch.save(student_agents[2].model.state_dict(), f"isvm3-00.pth")
    # torch.save(student_agents[3].model.state_dict(), f"isvm4-00.pth")
    # torch.save(student_agents[4].model.state_dict(), f"isvm5-00.pth")
    # torch.save(student_agents[5].model.state_dict(), f"isvm6-00.pth")
    # torch.save(student_agents[6].model.state_dict(), f"isvm7-00.pth")
    # torch.save(student_agents[7].model.state_dict(), f"isvm8-00.pth")

    # torch.save(student_agents[8].model.state_dict(), f"isvm9-00.pth")
    # torch.save(student_agents[9].model.state_dict(), f"isvm10-00.pth")
    # torch.save(student_agents[10].model.state_dict(), f"isvm11-00.pth")
    # torch.save(student_agents[11].model.state_dict(), f"isvm12-00.pth")
    for e in range(cfg.num_epochs):
        with ThreadPoolExecutor(max_workers=12) as executor:
            future_D0 = executor.submit(train_single_student, student_agents[0], rb_d, 0, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D1 = executor.submit(train_single_student, student_agents[1], rb_d, 1, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D2 = executor.submit(train_single_student, student_agents[2], rb_d, 2, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D3 = executor.submit(train_single_student, student_agents[3], rb_d, 3, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D4 = executor.submit(train_single_student, student_agents[4], rb_d, 4, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D5 = executor.submit(train_single_student, student_agents[5], rb_d, 5, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D6 = executor.submit(train_single_student, student_agents[6], rb_d, 6, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D7 = executor.submit(train_single_student, student_agents[7], rb_d, 7, cfg.model_config["window_size_s"], device=cfg.train_device)

            future_D8 = executor.submit(train_single_student, student_agents[8], rb_d, 8, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D9 = executor.submit(train_single_student, student_agents[9], rb_d, 9, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D10 = executor.submit(train_single_student, student_agents[10], rb_d, 10, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D11 = executor.submit(train_single_student, student_agents[11], rb_d, 11, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D12 = executor.submit(train_single_student, student_agents[12], rb_d, 12, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D13 = executor.submit(train_single_student, student_agents[13], rb_d, 13, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D14 = executor.submit(train_single_student, student_agents[14], rb_d, 14, cfg.model_config["window_size_s"], device=cfg.train_device)
            future_D15 = executor.submit(train_single_student, student_agents[15], rb_d, 15, cfg.model_config["window_size_s"], device=cfg.train_device)

            l0, g0, isvm_l0, l2_l0, ent_l0 = future_D0.result()
            l1, g1, isvm_l1, l2_l1, ent_l1 = future_D1.result()
            l2, g2, isvm_l2, l2_l2, ent_l2 = future_D2.result()
            l3, g3, isvm_l3, l2_l3, ent_l3 = future_D3.result()
            l4, g4, isvm_l4, l2_l4, ent_l4 = future_D4.result()
            l5, g5, isvm_l5, l2_l5, ent_l5 = future_D5.result()
            l6, g6, isvm_l6, l2_l6, ent_l6 = future_D6.result()
            l7, g7, isvm_l7, l2_l7, ent_l7 = future_D7.result()

            l8, g8, isvm_l8, l2_l8, ent_l8 = future_D8.result()
            l9, g9, isvm_l9, l2_l9, ent_l9 = future_D9.result()
            l10, g10, isvm_l10, l2_l10, ent_l10 = future_D10.result()
            l11, g11, isvm_l11, l2_l11, ent_l11 = future_D11.result()
            l12, g12, isvm_l12, l2_l12, ent_l12 = future_D12.result()
            l13, g13, isvm_l13, l2_l13, ent_l13 = future_D13.result()
            l14, g14, isvm_l14, l2_l14, ent_l14 = future_D14.result()
            l15, g15, isvm_l15, l2_l15, ent_l15 = future_D15.result()

            loss = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15]
            l1_loss = [isvm_l0, isvm_l1, isvm_l2, isvm_l3, isvm_l4, isvm_l5, isvm_l6, isvm_l7, isvm_l8, 
                        isvm_l9, isvm_l10, isvm_l11, isvm_l12, isvm_l13, isvm_l14, isvm_l15]
            l2_loss = [l2_l0, l2_l1, l2_l2, l2_l3, l2_l4, l2_l5, l2_l6, l2_l7, l2_l8, l2_l9, l2_l10, 
                        l2_l11, l2_l12, l2_l13, l2_l14, l2_l15]
            ent_loss = [ent_l0, ent_l1, ent_l2, ent_l3, ent_l4, ent_l5, ent_l6, ent_l7, ent_l8, ent_l9, 
                        ent_l10, ent_l11, ent_l12, ent_l13, ent_l14, ent_l15]
            grad = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15]

        print("train is over for now!!!")
        
        loss_stats = {
                "loss1": loss[0],
                "loss2": loss[1],
                "loss3": loss[2],
                "loss4": loss[3],
                "loss5": loss[4],
                "loss6": loss[5],
                "loss7": loss[6],
                "loss8": loss[7],
                "loss9": loss[8],
                "loss10": loss[9],
                "loss11": loss[10],
                "loss12": loss[11],
                "loss13": loss[12],
                "loss14": loss[13],
                "loss15": loss[14],
                "loss16": loss[15]
        }
        l1_stats = {
                "l1_loss1": l1_loss[0],
                "l1_loss2": l1_loss[1],
                "l1_loss3": l1_loss[2],
                "l1_loss4": l1_loss[3],
                "l1_loss5": l1_loss[4],
                "l1_loss6": l1_loss[5],
                "l1_loss7": l1_loss[6],
                "l1_loss8": l1_loss[7],
                "l1_loss9": l1_loss[8],
                "l1_loss10": l1_loss[9],
                "l1_loss11": l1_loss[10],
                "l1_loss12": l1_loss[11],
                "l1_loss13": l1_loss[12],
                "l1_loss14": l1_loss[13],
                "l1_loss15": l1_loss[14],
                "l1_loss16": l1_loss[15]
        }
        l2_stats = {
                "l2_loss1": l2_loss[0],
                "l2_loss2": l2_loss[1],
                "l2_loss3": l2_loss[2],
                "l2_loss4": l2_loss[3],
                "l2_loss5": l2_loss[4],
                "l2_loss6": l2_loss[5],
                "l2_loss7": l2_loss[6],
                "l2_loss8": l2_loss[7],
                "l2_loss9": l2_loss[8],
                "l2_loss10": l2_loss[9],
                "l2_loss11": l2_loss[10],
                "l2_loss12": l2_loss[11],
                "l2_loss13": l2_loss[12],
                "l2_loss14": l2_loss[13],
                "l2_loss15": l2_loss[14],
                "l2_loss16": l2_loss[15]
        }
        cross_entropy_stats = {
                "ent_loss1": ent_loss[0],
                "ent_loss2": ent_loss[1],
                "ent_loss3": ent_loss[2],
                "ent_loss4": ent_loss[3],
                "ent_loss5": ent_loss[4],
                "ent_loss6": ent_loss[5],
                "ent_loss7": ent_loss[6],
                "ent_loss8": ent_loss[7],
                "ent_loss9": ent_loss[8],
                "ent_loss10": ent_loss[9],
                "ent_loss11": ent_loss[10],
                "ent_loss12": ent_loss[11],
                "ent_loss13": ent_loss[12],
                "ent_loss14": ent_loss[13],
                "ent_loss15": ent_loss[14],
                "ent_loss16": ent_loss[15]
        }
        grad_stats = {
                "grad1": grad[0],
                "grad2": grad[1],
                "grad3": grad[2],
                "grad4": grad[3],
                "grad5": grad[4],
                "grad6": grad[5],
                "grad7": grad[6],
                "grad8": grad[7],
                "grad9": grad[8],
                "grad10": grad[9],
                "grad11": grad[10],
                "grad12": grad[11],
                "grad13": grad[12],
                "grad14": grad[13],
                "grad15": grad[14],
                "grad16": grad[15]
        }
        if e > 14499:
            torch.save(student_agents[0].model.state_dict(), f"isvm1-{e}.pth")
            torch.save(student_agents[1].model.state_dict(), f"isvm2-{e}.pth")
            torch.save(student_agents[2].model.state_dict(), f"isvm3-{e}.pth")
            torch.save(student_agents[3].model.state_dict(), f"isvm4-{e}.pth")
            torch.save(student_agents[4].model.state_dict(), f"isvm5-{e}.pth")
            torch.save(student_agents[5].model.state_dict(), f"isvm6-{e}.pth")
            torch.save(student_agents[6].model.state_dict(), f"isvm7-{e}.pth")
            torch.save(student_agents[7].model.state_dict(), f"isvm8-{e}.pth")

            torch.save(student_agents[8].model.state_dict(), f"isvm9-{e}.pth")
            torch.save(student_agents[9].model.state_dict(), f"isvm10-{e}.pth")
            torch.save(student_agents[10].model.state_dict(), f"isvm11-{e}.pth")
            torch.save(student_agents[11].model.state_dict(), f"isvm12-{e}.pth")
            torch.save(student_agents[12].model.state_dict(), f"isvm13-{e}.pth")
            torch.save(student_agents[13].model.state_dict(), f"isvm14-{e}.pth")
            torch.save(student_agents[14].model.state_dict(), f"isvm15-{e}.pth")
            torch.save(student_agents[15].model.state_dict(), f"isvm16-{e}.pth")

        metrics_student = StatsDict()
        metrics_student.extend(loss_stats)
        metrics_student.extend(grad_stats)
        metrics_student.extend(l1_stats)
        metrics_student.extend(l2_stats)
        metrics_student.extend(cross_entropy_stats)
        wandb_logger.log_student(metrics_student, metrics_student)

if __name__ == "__main__":
    # mp.set_start_method("spawn")
    main()
