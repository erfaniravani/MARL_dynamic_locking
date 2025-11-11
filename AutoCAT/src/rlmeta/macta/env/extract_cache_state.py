from collections import deque
import hashlib
import numpy as np
import random
import os
import yaml, logging
import sys
from itertools import permutations
import json
from typing import Optional

import gym
from gym import spaces #this works for Rlmeta
# from gymnasium.spaces import Discrete, Box #this works for torchRL

from omegaconf.omegaconf import open_dict
sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cache_simulator import *
import replacement_policy

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

        # table = UnixTable(sets)
        # table.title = cache.name
        # table.inner_row_border = True
        # table_lines = table.table
        max_col_widths = [max(len(item) for item in col) for col in zip(*sets)]
        # Construct the formatted table string
        table_string = ""
        for row in sets:
          formatted_row = [item.rjust(width) for item, width in zip(row, max_col_widths)]
          table_string += "|" + "|".join(formatted_row) + "|\n"

        # Add a separator line after the header
        table_string = table_string + "-" * (sum(max_col_widths) + len(max_col_widths) + 1) + "\n"
        return table_string


def load_trace(trace_file: str,
               limit: Optional[int] = None,
               legacy_trace_format: bool = False):
    with open(trace_file, "r") as f:
        lines = f.readlines()
        if limit is not None:
            lines = lines[:limit]

        data = []
        for line in lines:
            tokens = line.split()
            if legacy_trace_format:
                data.append(tokens)
            else:
                data.append((int(tokens[0]), int(tokens[3], 16)))

        if not legacy_trace_format:
            data = np.asarray(data, dtype=np.int64)

    return data

env_config={
    "length_violation_reward":-10000,
    "double_victim_access_reward": -10000,
    "force_victim_hit": False,
    "victim_access_reward":-10,
    "correct_reward":200,
    "wrong_reward":-9999,
    "step_reward":-1,
    "window_size":0,
    "attacker_addr_s":4,
    "attacker_addr_e":7,
    "victim_addr_s":0,
    "victim_addr_e":3,
    "flush_inst": False,
    "allow_victim_multi_access": True,
    "verbose":0,
    "reset_limit": 1,    # specify how many reset to end an epoch?????
    "cache_configs": {
        # YAML config file for cache simulaton
        "architecture": {
            "word_size": 1, #bytes
            "block_size": 1, #bytes
            "write_back": True
        },
        "cache_1": {#required
            "blocks": 16,
            "sets" : 2048, 
            "associativity": 16,  
            "hit_time": 1, #cycles
            "rep_policy": "new_plru_pl"
        },
        "mem": {#required
            "hit_time": 1000 #cycles
        }
        }
    }


def create_cache(trace_addr, output_dir):
    logger = logging.getLogger()
    fh = logging.FileHandler('log')
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    fh_format = logging.Formatter('%(message)s')
    fh.setFormatter(fh_format)
    sh.setFormatter(fh_format)
    logger.setLevel(logging.INFO)
    spec_trace = load_trace(trace_addr,
                    limit=60000000,
                    legacy_trace_format=False)
    configs = env_config["cache_configs"]
    hierarchy = build_hierarchy(configs, logger)
    l1 = hierarchy['cache_1']
    print("number of sets: ", l1.n_sets)
    print("trace address: ", trace_addr)
    hit_count = 0
    # Create output file name based on input file name and save it in the same directory
    output_file = os.path.join(output_dir, trace_addr.split('/')[-1].split('.')[0] + "_data.txt")
    
    # Open the output file for writing the formatted cache data
    with open(output_file, "w") as file:
        for current_step in range(len(spec_trace)):
            t, evict_addr, [set_index, way_index] = l1.read(hex(spec_trace[current_step][1] // 64)[2:], current_step)
            t = t.time
            hit_count += 1

            # Get cache data
            cache_data = l1.data['0']
            formatted_cache = []

            # Format the cache data
            for address, block in cache_data:
                if address == '--------':
                    formatted_cache.append('----')
                else:
                    # Convert binary address to hexadecimal
                    # hex_address = hex(int(address, 2)) #data look like 0x4f54f
                    hex_address = hex(int(address, 2))[2:] # data look like 4f54f
                    formatted_cache.append(hex_address)

            # Write formatted cache to file, each cache on a new line
            file.write(" ".join(formatted_cache) + "\n")

    return hit_count


def process_directory(directory_path):
    # Iterate through each file in the directory
    total_count = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it is a file and ends with a specific extension (e.g., '.trace')
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            print(f"Processing file: {file_path}")
            count = create_cache(file_path, directory_path)
            total_count += count
    print("total number of lines processed: ", total_count)

if __name__ == "__main__":
    dir_path = ""

    process_directory(dir_path)