from typing import Optional, List, Tuple
import gzip
import numpy as np


def load_trace(trace_file: str,
               limit: Optional[int] = None,
               legacy_trace_format: bool = False):
    with open(trace_file, "r") as f:
        lines = f.readlines()
        if limit is not None:
            lines = lines[:limit]

        data = []
        cache_state = []
        for line in lines:
            tokens = line.split()
            if legacy_trace_format:
                data.append(tokens)
            else:
                data.append((int(tokens[0]), int(tokens[3], 16)))
                last_8_elements = tokens[-8:]
                converted_elements = []
                for element in last_8_elements:
                    if element.startswith('0x'):
                        # Convert hex string to integer
                        converted_elements.append(int(element, 16))
                cache_state.append(converted_elements)

        if not legacy_trace_format:
            data = np.asarray(data, dtype=np.int64)
            # cache_state = np.asarray(cache_state, dtype=np.int64)

    return data, cache_state


