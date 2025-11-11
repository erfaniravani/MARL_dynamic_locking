# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import sys
import time
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, RMSRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_dict import StatsDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trace_parser import load_trace

console = Console()


class SpecAgent(Agent):
    def __init__(self, env_config, range_info_dict, trace, cache_data, legacy_trace_format: bool = False):
        super().__init__()

        self.local_step = 0
        self.lat = []
        self.no_prime = False  # set to true after first prime
        self.range_info_dict = range_info_dict
        self.prefill_count = 0

        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity']
            self.cache_size = self.configs['cache_1']['blocks']
            self.reset_benign_index_freq = env_config.get('reset_benign_index_freq', 0)
            self.reset_benign_index_counter = 0

            # attacker_addr_s = env_config[
            #     "attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            # attacker_addr_e = env_config[
            #     "attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            # victim_addr_s = env_config[
            #     "victim_addr_s"] if "victim_addr_s" in env_config else 0
            # victim_addr_e = env_config[
            #     "victim_addr_e"] if "victim_addr_e" in env_config else 3
            # flush_inst = env_config[
            #     "flush_inst"] if "flush_inst" in env_config else False
            # self.allow_empty_victim_access = env_config[
            #     "allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False

            attacker_addr_s = env_config.get("attacker_addr_s", 4)
            attacker_addr_e = env_config.get("attacker_addr_e", 7)
            victim_addr_s = env_config.get("victim_addr_s", 0)
            victim_addr_e = env_config.get("victim_addr_e", 3)
            flush_inst = env_config.get("flush_inst", False)
            self.attacker_address_space = attacker_addr_e - attacker_addr_s + 1

            self.allow_empty_victim_access = env_config.get(
                "allow_empty_victim_access", False)
            
            # print("************************************************************************")
            # print("inside the spec agent")

            # assert (self.num_ways == 1
            #         )  # currently only support direct-map cache
            assert (flush_inst == False)  # do not allow flush instruction
            # assert (attacker_addr_e - attacker_addr_s == victim_addr_e -
            #         victim_addr_s)  # address space must be shared
            #must be no shared address space
            # assert ((attacker_addr_e + 1 == victim_addr_s)
            #         or (victim_addr_e + 1 == attacker_addr_s))
            # assert (self.allow_empty_victim_access == False)

        # self.cache_line_size = 8  #TODO: remove the hardcode
        self.cache_line_size = env_config.get("cache_line_size", 64)
        self.trace = trace
        self.cache_data = cache_data
        self.trace_length = (len(self.trace)
                             if legacy_trace_format else self.trace.shape[0])
        assert isinstance(self.trace,
                          (list if legacy_trace_format else np.ndarray))
        self.legacy_trace_format = legacy_trace_format

      
        self._get_domain_ids()
        assert isinstance(self.domain_id_0, (int, np.int64))
        assert isinstance(self.domain_id_1, (int, np.int64))
        assert isinstance(self.domain_id_2, (int, np.int64))
        assert isinstance(self.domain_id_3, (int, np.int64))
        assert isinstance(self.domain_id_4, (int, np.int64))
        assert isinstance(self.domain_id_5, (int, np.int64))
        assert isinstance(self.domain_id_6, (int, np.int64))
        assert isinstance(self.domain_id_7, (int, np.int64))
        # print('these are the domain IDs: ', self.domain_id_0, self.domain_id_1, self.domain_id_2, self.domain_id_3, 
        #             self.domain_id_4, self.domain_id_5, self.domain_id_6, self.domain_id_7)

     

        random_key = random.choice(list(self.range_info_dict.keys()))
        start_line, end_line = self.range_info_dict[random_key]
        self.start_idx = random.randint(start_line, end_line)
        # self.start_idx = random.randint(0, self.trace_length - 200) #we dont want to be close to the end and roll over sicne the locality is differet
        self.step = 1#0

        # print(f"[Agent] cache_line_size = {self.cache_line_size}")

    def act(self, timestep: Optional[TimeStep] = None) -> Action:
        idx = (self.start_idx + self.step) % self.trace_length
        self.step = (self.step + 1) % self.trace_length
        # print("spec agent index: ", idx)
        if self.legacy_trace_format:
            line = self.trace[idx]
            domain_id = int(line[0])
            addr = int(line[3], 16) // self.cache_line_size
        else:
            domain_id, addr = self.trace[idx]
            addr //= self.cache_line_size

        assert isinstance(domain_id, (int, np.int64))
        assert isinstance(addr, (int, np.int64))

        action = addr % (self.cache_size*2)
        benign_addr = addr
        if domain_id == self.domain_id_0:  # attacker access
            action = benign_addr#self.trace[idx][1]#addr % (self.cache_size) # action encoding ---> does not have encodeing(victim_min). it seems to be fine
            info = {'d_id': 0}
        else:  # domain_id = self.domain_id_1: # victim access
            action = self.attacker_address_space
            addr = addr % self.cache_size # add victim_min
            if domain_id == self.domain_id_1:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 1}
            if domain_id == self.domain_id_2:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 2}
            if domain_id == self.domain_id_3:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 3}
            if domain_id == self.domain_id_4:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 4}
            if domain_id == self.domain_id_5:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 5}
            if domain_id == self.domain_id_6:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 6}
            if domain_id == self.domain_id_7:
                info = {"reset_victim_addr": True, "victim_addr": benign_addr, "d_id": 7}
            
        # print("^^^^^^^^^^", self.domain_id_0, domain_id, action, addr, self.trace[idx], idx)
        return Action(action, info)

    async def async_act(self, timestep: TimeStep) -> Action:
        return self.act(timestep)

    async def async_observe_init(self, timestep: TimeStep) -> None:
        pass

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        pass

    def update(self) -> None:
        pass

    async def async_update(self) -> None:
        pass

    def _get_domain_ids(self) -> None:
        # Read the first domain ID from the trace
        first_id = (
            int(self.trace[0][0]) if self.legacy_trace_format else self.trace[0, 0]
        )

        # Initialize a list with the first ID; we'll collect up to four unique IDs
        unique_ids = [first_id]

        # Iterate over the trace until we collect four distinct IDs (or exhaust the trace)
        for i in range(self.trace_length):
            cur = (
                int(self.trace[i][0])
                if self.legacy_trace_format
                else self.trace[i, 0]
            )
            if cur not in unique_ids:
                unique_ids.append(cur)
                if len(unique_ids) == 8:
                    break

        # If fewer than four unique IDs were found, pad with the first_id
        while len(unique_ids) < 8:
            unique_ids.append(first_id)

        # Assign them to self.domain_id_0 through self.domain_id_3
        self.domain_id_0 = unique_ids[0]
        self.domain_id_1 = unique_ids[1]
        self.domain_id_2 = unique_ids[2]
        self.domain_id_3 = unique_ids[3]
        self.domain_id_4 = unique_ids[4]
        self.domain_id_5 = unique_ids[5]
        self.domain_id_6 = unique_ids[6]
        self.domain_id_7 = unique_ids[7]

            
    
    def get_reset_data(self, mode):
        if self.reset_benign_index_counter == self.reset_benign_index_freq:
            # #for sample multiagent
            # self.rand_seed += 1
            # random.seed(self.rand_seed)
            # #for sample multiagent
            random_key = random.choice(list(self.range_info_dict.keys()))
            start_line, end_line = self.range_info_dict[random_key]
            self.start_idx = random.randint(start_line, end_line)

            self.reset_benign_index_counter = 1
            # self.prefill_count += 1

        elif self.reset_benign_index_counter < self.reset_benign_index_freq:
            self.reset_benign_index_counter += 1

        addresses = []
        if mode == "empty" or mode == 'env':
            return range(0, self.num_ways), self.reset_benign_index_counter
        else:
            idx = (self.start_idx + self.step) % self.trace_length
            self.step = (self.step + 1) % self.trace_length
            return self.cache_data[idx], self.reset_benign_index_counter #addresses

    def get_warmup_data(self):
        idx = (self.start_idx + self.step) % self.trace_length
        self.step = (self.step + 1) % self.trace_length
        if self.legacy_trace_format:
            line = self.trace[idx]
            domain_id = int(line[0])
            addr = int(line[3], 16) // self.cache_line_size
        else:
            domain_id, addr = self.trace[idx]
            addr //= self.cache_line_size

        assert isinstance(domain_id, (int, np.int64))
        assert isinstance(addr, (int, np.int64))
        benign_addr = addr
        if domain_id == self.domain_id_0:  # attacker access
            action = benign_addr#self.trace[idx][1]#addr % (self.cache_size) # action encoding ---> does not have encodeing(victim_min). it seems to be fine
            info = 0
        else:  # domain_id = self.domain_id_1: # victim access
            action = benign_addr
            info = 1
        return action, info, idx#, self.trace[idx]


class SpecAgentFactory(AgentFactory):
    def __init__(self,
                 env_config: Dict[str, Any],
                 trace_files: Sequence[str],
                 trace_limit: int,
                 legacy_trace_format: bool = False) -> None:
        self.env_config = env_config
        self.trace_files = trace_files
        self.trace_limit = trace_limit
        self.legacy_trace_format = legacy_trace_format


    def __call__(self, index: int) -> SpecAgent:
        spec_trace, cache_data, range_info_dict = self._load_trace(index)
        return SpecAgent(self.env_config,
                         range_info_dict,
                         spec_trace,
                         cache_data,
                         legacy_trace_format=self.legacy_trace_format)

    def parse_range_file(self, range_file):
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
                offset = 70 * (self.env_config.get('reset_benign_index_freq', 0)+1)
                range_dict[mod_key] = (start_line-1, end_line-offset)
        return range_dict

    def _load_trace(self, index: int) -> np.ndarray:
        trace_file = self.trace_files[index % len(self.trace_files)]
        root, ext = os.path.splitext(trace_file)
        range_file = root + '_ranges' + ext
        range_info_dict = self.parse_range_file(range_file)

        print(f"[SpecAgentFactory] agent [{index}] load {trace_file}")
        # print(range_info_dict)

        spec_trace, cache_data = load_trace(trace_file,
                                limit=self.trace_limit,
                                legacy_trace_format=self.legacy_trace_format)
        return spec_trace, cache_data, range_info_dict
