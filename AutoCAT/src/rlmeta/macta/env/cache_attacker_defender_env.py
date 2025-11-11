import os
import copy
import sys
import time
from ruamel.yaml import YAML
import yaml
from typing import Any, Dict, Sequence, Tuple
from collections import deque
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import hydra
import gym
from gym import spaces
from .cache_guessing_game_env import CacheGuessingGameEnv

from omegaconf.omegaconf import open_dict
sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cache_simulator import *
import replacement_policy
from agent import SpecAgent
from utils.trace_parser import load_trace



class CacheAttackerDefenderEnv(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True,
                 ) -> None:
        #env_config["cache_state_reset"] = False

        self.configs = env_config["cache_configs"]
        self.num_ways = self.configs['cache_1']['associativity']
        self.cache_sets = self.configs['cache_1']['sets']
        self.window_size = env_config.get("defender_window_size", 64)

        self.defender_correct_reward = env_config.get("defender_correct_reward", 10.0) 
        self.defender_wrong_reward = env_config.get("defender_wrong_reward", -10.0)
        self.hit_reward = env_config.get("hit_reward", 0.1)

        self.reset_mode = env_config.get("reset_mode", "trace")
        self.reset_warmup_length = env_config.get("reset_warmup_length", 64)

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        # self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        # self.action_space = spaces.Discrete(2**self._env.cache_size)
        #ToDO self.defender_action_space = 
        # self.defender_action_space = spaces.Discrete(2**self._env.cache_size)
        self.max_box_value = max(2**self._env.cache_size, self._env.max_box_value)#max(self.window_size + 2, len(self.attacker_address_space) + 1) 
        self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self.window_size, 3))



        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 64
        # print("this is max step from attack def env: ", self.max_step)
        self.detector_obs = deque([[-1, -1, -1]] * self.window_size)
        self.random_domain = random.choice([0,1])
        self.lst = [0, 1, 2, 3, 4, 5, 6, 7]
        random.shuffle(self.lst)
        self.first_half  = self.lst[:6]
        self.second_half = self.lst[6:]

        self.detector_reward_scale = 1.0 #0.1
        self.reset_data = range(0, self.num_ways)

        
        # self.current_benign_agent = random.choice(self.spec_agents_list)
        self.turn = 0
        # if "cache_configs" in env_config:
        #     self.configs = env_config["cache_configs"]
        # else:
        #     self.config_file_name = os.path.dirname(os.path.abspath(__file__))+'/../configs/config_simple_L1'
        #     self.config_file = open(self.config_file_name)
        #     self.logger.info('Loading config from file ' + self.config_file_name)
        #     self.configs = yaml.load(self.config_file, yaml.CLoader)
        # self.vprint(self.configs)

    def reset(self, reset_data= None, reset_observation=False, victim_address=-1):
        """
        returned obs = { agent_name : obs }
        """
        # print("printing to check the reset data:  ", reset_data)
        if reset_data:
            self.reset_data = reset_data
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        if self.opponent_agent == 'benign':
            self.ben = 1
        else:
            self.ben = 0
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.fake_count = 0
        current = time.time()
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True, 
                                       reset_benign= self.opponent_agent=='benign',
                                       reset_data= self.reset_data,
                                       seed = int(current))
        self.victim_address = self._env.victim_address
        if reset_observation:
            self.detector_obs = deque([[-1, -1, -1]] * self.window_size)
        self.random_domain = random.choice([0,1])
        self.lst = [0, 1, 2, 3, 4, 5, 6, 7]
        random.shuffle(self.lst)
        self.first_half  = self.lst[:6] 
        self.second_half = self.lst[6:]
        # self.current_benign_agent = random.choice(self.spec_agents_list)
        self.turn = 0
        # print("reset method in attacker defender env  ", len(reset_data), self.opponent_agent, self.victim_address)
        obs = {}
        obs['defender'] = np.array(list(reversed(self.detector_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs#np.array([self.benign_trace_idx, self.ben])
        return obs
    
    def get_detector_obs(self, opponent_obs, opponent_info, action_info):
        skip = 0
        if self.opponent_agent == 'attacker' and self.turn == 0:
            cur_opponent_obs = self._env.latest_observation
        else:
            cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        cur_obs = [cur_opponent_obs[0],-1,-1]
        if not np.any(cur_opponent_obs==-1):
            if not self._env.is_empty_access:
                if opponent_info.get('invoke_victim'):
                    cur_obs[0] = opponent_info['victim_latency']
                    cur_obs[1] = self.random_domain #1
                else:
                    cur_obs[1] = 1-self.random_domain#0
                cur_obs[2] = opponent_info['way_index']
                # cur_obs[3] = opponent_info['set_index']
                if self.opponent_agent == "benign":
                    cur_obs[1] = self.lst[action_info.get("d_id")]
                    # print("here0 ", cur_obs[1], self.lst, action_info)
                if self.opponent_agent == 'attacker' and self.turn == 0:
                    # print("this should not go out of range: ", action_info.get("d_id"))
                    if action_info.get("d_id") < 6:
                        cur_obs[1] = self.first_half[action_info.get("d_id")]
                    else:
                        self.turn = 1
                        skip = 1
                    
                    # print("here ", cur_obs[1], self.lst, action_info, self.first_half, self.second_half)
                if self.opponent_agent == 'attacker' and self.turn == 1:
                    index = cur_obs[1]
                    cur_obs[1] = self.second_half[index]
                    # print("here2 ", cur_obs[1], self.lst, action_info, self.first_half, self.second_half)
                if skip == 0:
                    self.detector_obs.append(cur_obs)
                    self.detector_obs.popleft()
        return np.array(list(reversed(self.detector_obs)))
        # return self.detector_obs
        # return cur_opponent_obs

    def compute_reward(self, action, reward, opponent_done, opponent_info, opponent_attack_success=False):
        # detector_flag = False
        detector_correct = False
        detector_reward = 0
        benign_hit_reward = 0.000000000001
        #give positive reward for benign hit 
        if self.opponent_agent == 'benign':
            if opponent_info["hit_miss"] == "hit":
                benign_hit_reward += self.hit_reward
            elif opponent_info["hit_miss"] == "miss":
                benign_hit_reward += -self.hit_reward
        
        if self.opponent_agent == 'attacker' and self.turn == 0:
            if opponent_info["hit_miss"] == "hit":
                benign_hit_reward += self.hit_reward
            elif opponent_info["hit_miss"] == "miss":
                benign_hit_reward += -self.hit_reward

        if self.opponent_agent == 'attacker' and self.turn and opponent_done and opponent_attack_success:
            # attacker episode terminate, and has attacked *successfully*
            detector_reward = self.defender_wrong_reward
        elif self.opponent_agent == 'attacker' and self.turn and opponent_done:
            # attacker episode terminates, but has done nothing successfully
            detector_reward = self.defender_correct_reward
        
        attacker_reward = reward['attacker']

        rew = {}
        rew['defender'] = (detector_reward + benign_hit_reward) * self.detector_reward_scale
        rew['attacker'] = attacker_reward

        info = {}
        info['guess_correct'] = detector_correct
        return rew, info

    def fake(self, action, info):
        self.fake_count += 1
        obs = {}
        if action["defender"] > 65535 :
            detector_action = '0000000000000000'
        else:
            bin_rep = format(int(action["defender"]), '016b') #need only this for the correct attack-defender  
            detector_action = str(bin_rep)
        self._env.l1.detector_func(detector_action)
        
        benign_domain_id = info
        t, evict_addr, [set_index, way_index] = self._env.l1.read(hex(action['benign'])[2:], self.fake_count)
        set_index = int(set_index,2)
        t = t.time
        if t > 500:
            r = 1 #miss
        else:
            r = 0
        self.detector_obs.append([r, benign_domain_id, way_index])
        self.detector_obs.popleft()
        # self._env.print_sample_multiagent_cache()
        return np.array(list(reversed(self.detector_obs)))


    def step(self, action):
        self._env.is_fake_benign = False
        self.step_count += 1
        obs = {}
        reward = {}
        done = {'__all__':False}
        info = {}
        action_info = action.get('info')
        # print("looking into the action  ", action, self.opponent_agent)
        # print("at every step, defender: ", action["defender"], " attacker: ", action["attacker"], "  benign: ", action['benign'])
        # print(action_info)
        if action["defender"] > 65535 :
            detector_action = '0000000000000000'
        else:
            bin_rep = format(int(action["defender"]), '016b') #need only this for the correct attack-defender  
            detector_action = str(bin_rep)
        self._env.l1.detector_func(detector_action)
        if self.opponent_agent == 'benign':
            self._env.is_benign = True
        else:
            self._env.is_benign = False
        self._env.benign_reset_victim_add = None
        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            # self.benign_trace_idx = action_info.get('trace_idx', -1)
            if self.opponent_agent == 'benign' and benign_reset_victim:
                # self._env.set_victim(benign_victim_addr) 
                self._env.benign_reset_victim_add = benign_victim_addr
                self.victim_address = self._env.victim_address
                # print("********", benign_victim_addr, self.victim_address)

        if self.opponent_agent == 'benign':
            opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
            opponent_info["opponent"] = self.opponent_agent
            # opponent_info["set"] = self.current_set
            # print("final attacker's observation: ", opponent_obs[0])
            if opponent_done:
                current = time.time()
                opponent_obs = self._env.reset(reset_cache_state=True, seed = current, reset_benign=self.opponent_agent=='benign', reset_data= self.reset_data)
                self.victim_address = self._env.victim_address
                self.step_count -= 1 # The reset/guess step should not be counted
            if self.step_count >= self.max_step:
                if self.opponent_agent == "benign":
                    cache_data = self._env.l1.data['0']
                    formatted_cache = []
                    # Format the cache data
                    for address, block in cache_data:
                        int_address = int(address, 2)
                        formatted_cache.append(int_address)
                    # print("is this the data i need?  ", formatted_cache)
                    self.reset_data = formatted_cache
                detector_done = True
                current = time.time()
                opponent_obs = self._env.reset(reset_cache_state=True, seed = current, reset_benign=self.opponent_agent=='benign', reset_data= self.reset_data)
                self.victim_address = self._env.victim_address
            else:
                detector_done = False
            # attacker
            obs['attacker'] = opponent_obs
            reward['attacker'] = opponent_reward
            done['attacker'] = detector_done #Figure out correctness
            info['attacker'] = opponent_info
            
            #benign
            obs['benign'] = opponent_obs#np.array([self.benign_trace_idx, self.ben])
            reward['benign'] = opponent_reward
            done['benign'] = detector_done #Figure out correctness
            info['benign'] = opponent_info
            opponent_attack_success = opponent_info.get('guess_correct', False)

            # obs, reward, done, info 
            updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, opponent_info, opponent_attack_success)
            reward['attacker'] = updated_reward['attacker']
            reward['defender'] = updated_reward['defender']
            # print('defender reward', updated_reward['defender'])
            # print('attacker reward', updated_reward['attacker'])

            obs['defender'] = self.get_detector_obs(opponent_obs, opponent_info, action_info) 
            done['defender'] = detector_done
            info['defender'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['defender'])}
            info['defender'].update(opponent_info)
            # Change the criteria to determine wether the game is done
            if detector_done:
                done['__all__'] = True
            #from IPython import embed; embed()

            info['__all__'] = {'action_mask':self.action_mask}
        
            for k,v in info.items():
                info[k].update({'action_mask':self.action_mask})
            # print("final defender's observation", obs["defender"])
            # print("final attacker's observation", obs["attacker"])
            # print_cache(self._env.l1)
            # self._env.print_sample_multiagent(obs, reward, done, info, self.opponent_agent, action["defender"])
            return obs, reward, done, info

        if self.opponent_agent == 'attacker':
            if self.turn == 0:
                self._env.is_fake_benign = True
                benign_action = action["benign"] # self.current_benign_agent.act()
                b_info = action_info # benign_action.info
                benign_reset_victim = b_info.get('reset_victim_addr', False)
                benign_victim_addr = b_info.get('victim_addr', None)
                if benign_reset_victim:
                    self._env.benign_reset_victim_add = benign_victim_addr
                    self.victim_address = self._env.victim_address
                # print('hip hip hoora', benign_action, action_info)
                # opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(benign_action.action)
                opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(benign_action)
                opponent_info["opponent"] = self.opponent_agent
                self._env.is_fake_benign = False

                if self.step_count >= self.max_step*2:
                    if self.opponent_agent == "benign":
                        cache_data = self._env.l1.data['0']
                        formatted_cache = []
                        # Format the cache data
                        for address, block in cache_data:
                            int_address = int(address, 2)
                            formatted_cache.append(int_address)
                        # print("is this the data i need?  ", formatted_cache)
                        self.reset_data = formatted_cache
                    detector_done = True
                    current = time.time()
                    opponent_obs = self._env.reset(reset_cache_state=True, seed = current, reset_benign=self.opponent_agent=='benign', reset_data= self.reset_data)
                    self.victim_address = self._env.victim_address
                else:
                    detector_done = False
                # attacker
                obs['attacker'] = opponent_obs
                reward['attacker'] = opponent_reward
                done['attacker'] = detector_done #Figure out correctness
                info['attacker'] = opponent_info
                
                #benign
                obs['benign'] = opponent_obs#np.array([self.benign_trace_idx, self.ben])
                reward['benign'] = opponent_reward
                done['benign'] = detector_done #Figure out correctness
                info['benign'] = opponent_info
                opponent_attack_success = opponent_info.get('guess_correct', False)

                # obs, reward, done, info 
                updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, opponent_info, opponent_attack_success)
                reward['attacker'] = updated_reward['attacker']
                reward['defender'] = updated_reward['defender']
                # print('defender reward', updated_reward['defender'])
                # print('attacker reward', updated_reward['attacker'])

                obs['defender'] = self.get_detector_obs(opponent_obs, opponent_info, b_info) 
                done['defender'] = detector_done
                info['defender'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['defender'])}
                info['defender'].update(opponent_info)
                # Change the criteria to determine wether the game is done
                if detector_done:
                    done['__all__'] = True
                #from IPython import embed; embed()

                info['__all__'] = {'action_mask':self.action_mask}
            
                for k,v in info.items():
                    info[k].update({'action_mask':self.action_mask})
                # print("final defender's observation", obs["defender"])
                # print("final attacker's observation", obs["attacker"])
                

            if self.turn == 1:
                opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
                opponent_info["opponent"] = self.opponent_agent
                # opponent_info["set"] = self.current_set
                # print("final attacker's observation: ", opponent_obs[0])
                if opponent_done:
                    current = time.time()
                    opponent_obs = self._env.reset(reset_cache_state=True, seed = current, reset_benign=self.opponent_agent=='benign', reset_data= self.reset_data)
                    self.victim_address = self._env.victim_address
                    self.step_count -= 1 # The reset/guess step should not be counted
                if self.step_count >= self.max_step*2:
                    if self.opponent_agent == "benign":
                        cache_data = self._env.l1.data['0']
                        formatted_cache = []
                        # Format the cache data
                        for address, block in cache_data:
                            int_address = int(address, 2)
                            formatted_cache.append(int_address)
                        # print("is this the data i need?  ", formatted_cache)
                        self.reset_data = formatted_cache
                    detector_done = True
                    current = time.time()
                    opponent_obs = self._env.reset(reset_cache_state=True, seed = current, reset_benign=self.opponent_agent=='benign', reset_data= self.reset_data)
                    self.victim_address = self._env.victim_address
                else:
                    detector_done = False
                # attacker
                obs['attacker'] = opponent_obs
                reward['attacker'] = opponent_reward
                done['attacker'] = detector_done #Figure out correctness
                info['attacker'] = opponent_info
                
                #benign
                obs['benign'] = opponent_obs#np.array([self.benign_trace_idx, self.ben])
                reward['benign'] = opponent_reward
                done['benign'] = detector_done #Figure out correctness
                info['benign'] = opponent_info
                opponent_attack_success = opponent_info.get('guess_correct', False)

                # obs, reward, done, info 
                updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, opponent_info, opponent_attack_success)
                reward['attacker'] = updated_reward['attacker']
                reward['defender'] = updated_reward['defender']
                # print('defender reward', updated_reward['defender'])
                # print('attacker reward', updated_reward['attacker'])

                obs['defender'] = self.get_detector_obs(opponent_obs, opponent_info, action_info) 
                done['defender'] = detector_done
                info['defender'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['defender'])}
                info['defender'].update(opponent_info)
                # Change the criteria to determine wether the game is done
                if detector_done:
                    done['__all__'] = True
                #from IPython import embed; embed()

                info['__all__'] = {'action_mask':self.action_mask}
            
                for k,v in info.items():
                    info[k].update({'action_mask':self.action_mask})
                # print("final defender's observation", obs["defender"])
                # print("final attacker's observation", obs["attacker"])
                # print_cache(self._env.l1)
                # self._env.print_sample_multiagent(obs, reward, done, info, self.opponent_agent, action["defender"])
            # print("current turn = ", self.turn, "  changing turn and submitting the obs and rewards...")
            self.turn = 1 - self.turn
            return obs, reward, done, info

@hydra.main(config_path="../config", config_name="macta")
def main(cfg):
    locked_lines1 = random.choices(['0','1'], weights=[0.3, 0.7], k=1)[0]
    locked_lines2 = random.choices(['0','1'], weights=[0.3, 0.7], k=1)[0]
    print(locked_lines1+locked_lines2, "    ", type(locked_lines1+locked_lines2))
    env = CacheAttackerDefenderEnv(cfg.env_config)
    env.opponent_weights = [0,1]
    action_space = env.action_space
    obs = env.reset()
    done = {'__all__':False}
    prev_a = 10
    test_action = format(random.randint(0, 15), '04b')
    bin_rep = format(15, '04b') 
    detector_action = str(bin_rep)
    print("check defender action: ", detector_action, type(detector_action))
    for k in range(1):
        i = 0
        while not done['__all__']:
            i += 1
            print("step: ", i)
            action = {'attacker':9 if (prev_a==10 or i<64) else 10, #np.random.randint(low=9, high=11),
                      'benign':np.random.randint(low=2, high=5),
                      'defender': np.random.randint(low=0, high=255)} #generate 8 bit random numbers to represent lock bits
            prev_a = action['attacker']
            print("actions:  ", action)
            obs, reward, done, info = env.step(action)
            #print("obs: ", obs['detector'])
            print("action: ", action)
            # print("victim: ", env.victim_address, env._env.victim_address)

    #         #print("done:", done)
    #         # print("reward:", reward)
    #         #print(env.victim_address_min, env.victim_address_max)
    #         #print("info:", info )
    #         if info['attacker'].get('invoke_victim') or info['attacker'].get('is_guess')==True:
    #             # print(info['attacker'])
    #             pass
    #     obs = env.reset()
    #     done = {'__all__':False}







def print_cache(cache):
    # Print the contents of a cache as a table
    # If the table is too long, it will print the first few sets,
    # break, and then print the last set
    table_size = 5
    ways = [""]
    sets = []
    set_indexes = sorted(cache.data.keys())
    
    if len(cache.data.keys()) > 0:
        way_no = 0

        # Label the columns
        for way in range(cache.associativity):
            ways.append("Way " + str(way_no))
            way_no += 1

        # Print either all the sets if the cache is small, or just a few
        # sets and then the last set
        sets.append(ways)
        if len(set_indexes) > table_size + 4 - 1:
            for s in range(min(table_size, len(set_indexes) - 4)):
                set_ways = cache.data[set_indexes[s]].keys()
                temp_way = ["Set " + str(s)]
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[s]][w].address)
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

            for i in range(3):
                temp_way = ['.']
                for w in range(cache.associativity):
                    temp_way.append('')
                sets.append(temp_way)

            set_ways = cache.data[set_indexes[len(set_indexes) - 1]].keys()
            temp_way = ['Set ' + str(len(set_indexes) - 1)]
            for w in range(0, cache.associativity):
                temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w][1].address)
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w].address)
            sets.append(temp_way)
            
        else:
            for s in range(len(set_indexes)):
                temp_way = ["Set " + str(s)]
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

                # add additional rows only if the replacement policy = lru_lock_policy
                if cache.rep_policy == lru_lock_policy:
                    lock_info = ["Lock bit"]

                    lock_vector_array = cache.set_rep_policy[set_indexes[s]].lock_vector_array

                    for w in range(0, len(lock_vector_array)):
                        lock_info.append(lock_vector_array[w])
                    sets.append(lock_info)

                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                    sets.append(timestamp)
                elif cache.rep_policy == new_plru_pl_policy: # add a new row to the table to show the lock bit in the plru_pl_policy cache
                    lock_info = ["Lock bit"]

                    lockarray = cache.set_rep_policy[set_indexes[s]].lockarray

                    for w in range(0, len(lockarray)):
                        if lockarray[w] == 2:
                            lock_info.append("unlocked")
                        elif lockarray[w] == 1:
                            lock_info.append("locked")
                        elif lockarray[w] == 0:
                            lock_info.append("unknown")
                        else:
                            lock_info.append(lockarray[w])
                    sets.append(lock_info)
                elif cache.rep_policy == lru_policy:  # or cache.rep_policy == lru_lock_policy:
                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                            
                    sets.append(timestamp)
                    # print(timestamp)

        table = UnixTable(sets)
        table.title = cache.name
        table.inner_row_border = True
        print(table.table)
        return set_indexes

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



if __name__ == "__main__":
    main()
