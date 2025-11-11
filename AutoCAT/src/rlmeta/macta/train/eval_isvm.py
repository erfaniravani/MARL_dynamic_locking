import os
import sys
import copy
import logging
import time
import subprocess
import platform
import psutil
import random

import hydra

import torch
import torch.multiprocessing as mp
import re
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
from env.cache_attacker_defender_env_factory import CacheAttackerDefenderEnvFactory

from utils.ma_metric_callbacks import MACallbacks
from utils.wandb_logger import WandbLogger, stats_filter
from utils.controller import Phase, Controller
from utils.test_maloop import LoopList, MAParallelLoop

from utils.trace_parser import load_trace

from agent import RandomAgent, BenignAgent, SpecAgent, PPOAgent
from agent import SpecAgentFactory

def prepare_model_pool(defender_model, attacker_model, cfg):
    directory = cfg.prepare_pool
    detector_pattern = re.compile(r'^detector-(\d+)\.pth$')
    attacker_pattern = re.compile(r'^attacker-(\d+)\.pth$')
    
    try:
        for filename in os.listdir(directory):
            detector_match = detector_pattern.match(filename)
            attacker_match = attacker_pattern.match(filename)
            
            if detector_match:
                number = int(detector_match.group(1))
                if number % 10 == 9:
                    full_path = os.path.join(directory, filename)
                    defender_add_pool = torch.load(full_path, map_location=cfg.train_device)
                    attacker_model.push_to_history(defender_add_pool)
            
            elif attacker_match:
                number = int(attacker_match.group(1))
                if number % 10 == 9:
                    full_path = os.path.join(directory, filename)
                    attacker_add_pool = torch.load(full_path, map_location=cfg.train_device)
                    attacker_model.push_to_history(attacker_add_pool)
    
    except FileNotFoundError:
        print(f"Directory {directory} not found.")
    return defender_model, attacker_model

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
    env = env_fac(0)
    #### attacker
    cfg.model_config["output_dim"] = env.action_space.n


    train_model = CachePPOTransformerModelPool(**cfg.model_config).to(
        cfg.train_device)
    attacker_checkpoint = cfg.attacker_checkpoint

    # if cfg.use_attacker_pool:
    #     attacker_pool_params1 = torch.load(cfg.attacker_pool1, map_location=cfg.train_device)
    #     train_model.push_to_history(attacker_pool_params1)
    #     attacker_pool_params2 = torch.load(cfg.attacker_pool2, map_location=cfg.train_device)
    #     train_model.push_to_history(attacker_pool_params2)
    #     attacker_pool_params3 = torch.load(cfg.attacker_pool3, map_location=cfg.train_device)
    #     train_model.push_to_history(attacker_pool_params3)
    #     attacker_pool_params4 = torch.load(cfg.attacker_pool4, map_location=cfg.train_device)
    #     train_model.push_to_history(attacker_pool_params4)
    #     attacker_pool_params5 = torch.load(cfg.attacker_pool5, map_location=cfg.train_device)
    #     train_model.push_to_history(attacker_pool_params5)

    if len(attacker_checkpoint) > 0:
        attacker_params = torch.load(cfg.attacker_checkpoint, map_location=cfg.train_device)
        train_model.load_state_dict(attacker_params)
    # train_model.eval()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.lr)

    infer_model = copy.deepcopy(train_model).to(cfg.infer_device)
    infer_model.eval()

    ctrl = Controller()
    rb = ReplayBuffer(cfg.replay_buffer_size)
    # #### detector
    cfg.model_config["output_dim"] = 2**8
    cfg.model_config["step_dim"] += 2
    train_model_d = CachePPOTransformerModelPoolDefender(**cfg.model_config).to(
        cfg.train_device_d)

    # if cfg.use_attacker_pool:
    #     train_model_d, train_model = prepare_model_pool(train_model_d, train_model, cfg)

    # if cfg.trained_defender:
    #     if cfg.env_config.new_arc:
    #         d_params = torch.load(cfg.new_detector_checkpoint, map_location=cfg.train_device)
    #         train_model_d.load_state_dict(d_params)
    #     else:
    #         d_params = torch.load(cfg.old_detector_checkpoint, map_location=cfg.train_device)
    #         train_model_d.load_state_dict(d_params)
    optimizer_d = torch.optim.Adam(train_model_d.parameters(), lr=cfg.lr_d)

    infer_model_d = copy.deepcopy(train_model_d).to(cfg.infer_device_d)
    infer_model_d.eval()

    rb_d = ReplayBuffer(cfg.replay_buffer_size_d)
    # =========================================================================
    
    #### start server
    # =========================================================================
    m_server = Server(cfg.m_server_name, cfg.m_server_addr)
    r_server = Server(cfg.r_server_name, cfg.r_server_addr)
    c_server = Server(cfg.c_server_name, cfg.c_server_addr)
    m_server.add_service(infer_model)
    r_server.add_service(rb)
    c_server.add_service(ctrl)
    md_server = Server(cfg.md_server_name, cfg.md_server_addr)
    rd_server = Server(cfg.rd_server_name, cfg.rd_server_addr)
    md_server.add_service(infer_model_d)
    rd_server.add_service(rb_d)
    servers = ServerList([m_server, r_server, c_server, md_server, rd_server])
    # =========================================================================

    #### Define remote model and control
    # =========================================================================
    a_model = wrap_downstream_model(train_model, m_server)
    t_model = remote_utils.make_remote(infer_model, m_server)
    ea_model = remote_utils.make_remote(infer_model, m_server)
    ed_model = remote_utils.make_remote(infer_model, m_server)
    td_model = remote_utils.make_remote(infer_model, m_server)
    # ---- control
    a_ctrl = remote_utils.make_remote(ctrl, c_server)
    ta_ctrl = remote_utils.make_remote(ctrl, c_server)
    td_ctrl = remote_utils.make_remote(ctrl, c_server)
    ea_ctrl = remote_utils.make_remote(ctrl, c_server)
    ed_ctrl = remote_utils.make_remote(ctrl, c_server)
    # =========================================================================

    a_rb = make_remote_replay_buffer(rb, r_server, prefetch=cfg.prefetch)
    t_rb = make_remote_replay_buffer(rb, r_server)

    agent = PPOAgent(a_model,
                     replay_buffer=a_rb,
                     controller=a_ctrl,
                     optimizer=optimizer,
                     batch_size=cfg.batch_size,
                     learning_starts=cfg.get("learning_starts", None),
                     entropy_coeff=cfg.get("entropy_coeff", 0.01),
                     push_every_n_steps=cfg.push_every_n_steps, 
                     gamma = cfg.get("gamma", 0.99))
    ta_agent_fac = AgentFactory(PPOAgent, t_model, replay_buffer=t_rb)
    td_agent_fac = AgentFactory(PPOAgent, td_model, deterministic_policy=True)
    ea_agent_fac = AgentFactory(PPOAgent, ea_model, deterministic_policy=True)
    ed_agent_fac = AgentFactory(PPOAgent, ed_model, deterministic_policy=True)
    #### random detector
    
    '''
    detector = RandomAgent(2)
    t_d_fac = AgentFactory(RandomAgent, 2)
    e_d_fac = AgentFactory(RandomAgent, 2)
    '''
    '''
    #### random benign agent
    benign = BenignAgent(env.action_space.n)
    t_b_fac = AgentFactory(BenignAgent, env.action_space.n)
    e_b_fac = AgentFactory(BenignAgent, env.action_space.n)
    #### spec benign agent
    
    '''
   
    t_b_fac = SpecAgentFactory(cfg.env_config,
                               cfg.trace_files,#collected_files
                               cfg.trace_limit,
                               legacy_trace_format=cfg.legacy_trace_format)
    e_b_fac = SpecAgentFactory(cfg.env_config,
                               cfg.trace_files,#collected_files,
                               cfg.trace_limit,
                               legacy_trace_format=cfg.legacy_trace_format)



    #### detector agent
    a_model_d = wrap_downstream_model(train_model_d, md_server)
    t_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ea_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ed_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ta_model_d = remote_utils.make_remote(infer_model_d, md_server)
    a_rb_d = make_remote_replay_buffer(rb_d, rd_server, prefetch=cfg.prefetch)
    t_rb_d = make_remote_replay_buffer(rb_d, rd_server)

    agent_d = PPOAgent(a_model_d,
                     replay_buffer=a_rb_d,
                     controller=a_ctrl,
                     optimizer=optimizer_d,
                     batch_size=cfg.batch_size_d,
                     learning_starts=cfg.get("learning_starts_d", None),
                     entropy_coeff=cfg.get("entropy_coeff_d", 0.01),
                     entropy_decay = False,
                     defender = cfg.env_config.new_arc,
                     push_every_n_steps=cfg.push_every_n_steps_d, 
                     gamma = cfg.get("gamma_d", 0.99),
                     eps_clip = cfg.get("eps_clip", 0.2))
    td_d_fac = AgentFactory(PPOAgent, t_model_d, replay_buffer=t_rb_d)
    ta_d_fac = AgentFactory(PPOAgent, ta_model_d, deterministic_policy=True)
    ea_d_fac = AgentFactory(PPOAgent, ea_model_d, deterministic_policy=True)
    ed_d_fac = AgentFactory(PPOAgent, ed_model_d, deterministic_policy=True)

    #### create agent list
    ta_ma_fac = {"benign":t_b_fac, "attacker":ta_agent_fac, "defender":ta_d_fac}
    td_ma_fac = {"benign":t_b_fac, "attacker":td_agent_fac, "defender":td_d_fac}
    ea_ma_fac = {"benign":e_b_fac, "attacker":ea_agent_fac, "defender":ea_d_fac}
    ed_ma_fac = {"benign":e_b_fac, "attacker":ed_agent_fac, "defender":ed_d_fac}

    ta_loop = MAParallelLoop(env_fac_unbalanced,
                          ta_ma_fac,
                          ta_ctrl,
                          running_phase=Phase.TRAIN_ATTACKER,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.train_seed,
                          episode_callbacks=my_callbacks, 
                          cfg=cfg)
    if cfg.def_train_env == "unbalanced":
        td_loop = MAParallelLoop(env_fac_unbalanced,
                          td_ma_fac,
                          td_ctrl,
                          running_phase=Phase.TRAIN_DETECTOR,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.train_seed,
                          episode_callbacks=my_callbacks,
                          cfg=cfg)
    elif cfg.def_train_env == "balanced":
        td_loop = MAParallelLoop(env_fac,
                            td_ma_fac,
                            td_ctrl,
                            running_phase=Phase.TRAIN_DETECTOR,
                            should_update=True,
                            num_rollouts=cfg.num_train_rollouts,
                            num_workers=cfg.num_train_workers,
                            seed=cfg.train_seed,
                            episode_callbacks=my_callbacks,
                            cfg=cfg)
    elif cfg.def_train_env == "benign":
        td_loop = MAParallelLoop(env_fac_benign,
                            td_ma_fac,
                            td_ctrl,
                            running_phase=Phase.TRAIN_DETECTOR,
                            should_update=True,
                            num_rollouts=cfg.num_train_rollouts,
                            num_workers=cfg.num_train_workers,
                            seed=cfg.train_seed,
                            episode_callbacks=my_callbacks,
                            cfg=cfg)
    ea_loop = MAParallelLoop(env_fac_unbalanced,
                          ea_ma_fac,
                          ea_ctrl,
                          running_phase=Phase.EVAL_ATTACKER,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=cfg.eval_seed,
                          episode_callbacks=my_callbacks,
                          cfg=cfg)
    ed_loop = MAParallelLoop(env_fac_benign,
                          ed_ma_fac,
                          ed_ctrl,
                          running_phase=Phase.EVAL_DETECTOR,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=cfg.eval_seed,
                          episode_callbacks=my_callbacks,
                          cfg=cfg)

    loops = LoopList([ta_loop, td_loop, ea_loop, ed_loop])

    servers.start()
    # time.sleep(120)
    loops.start()
    agent.connect()
    agent_d.connect()
    a_ctrl.connect()

    start_time = time.perf_counter()
    turn = 1
    attacker_count = 0
    defender_count = 0

    for epoch in range(cfg.num_epochs):

        a_stats, d_stats = None, None
        a_ctrl.set_phase(Phase.TRAIN, reset=True)

        print("agent being trained: Attacker, Epoch: ", epoch)
        is_attacker = 1
        agent_d.set_use_history(False)
        agent.set_use_history(False)
        agent.controller.set_phase(Phase.TRAIN_ATTACKER, reset=True)
        a_stats = agent.train(cfg.steps_per_epoch)
        #wandb_logger.save(epoch, train_model, prefix="attacker-")
        torch.save(train_model.state_dict(), f"attacker-{epoch}.pth")
        train_stats = {"attacker":a_stats}
        # if epoch % 10 == 9:
        #     agent.model.push_to_history()

        stats = a_stats or d_stats

        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))
        time.sleep(1)

        a_ctrl.set_phase(Phase.EVAL, limit=cfg.num_eval_episodes, reset=True)
        agent.set_use_history(False)
        agent_d.set_use_history(False)
        agent.controller.set_phase(Phase.EVAL_ATTACKER, limit=cfg.num_eval_episodes, reset=True)
        a_stats = agent.eval(cfg.num_eval_episodes)
        agent_d.controller.set_phase(Phase.EVAL_DETECTOR, limit=cfg.num_eval_episodes, reset=True)
        d_stats = agent_d.eval(cfg.num_eval_episodes)
        stats = a_stats
        agent_trained_stats = {
            "is_attacker": is_attacker
        }
        stats.extend(agent_trained_stats)

        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))
        eval_stats = {"attacker":a_stats, "detector":d_stats}
        time.sleep(1)
        
        wandb_logger.log(train_stats, eval_stats)
    


    loops.terminate()
    servers.terminate()
    


def add_prefix(input_dict, prefix=''):
    res = {}
    for k,v in input_dict.items():
        res[prefix+str(k)]=v
    return res

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
