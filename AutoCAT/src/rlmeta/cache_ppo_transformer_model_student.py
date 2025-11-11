import os
import sys

from typing import Dict, List, Tuple, Optional, Union

import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote
import rlmeta.utils.nested_utils as nested_utils
from rlmeta.agents.ppo.ppo_model import PPOModel
from rlmeta.core.model import DownstreamModel, RemotableModel
from rlmeta.core.server import Server

class CachePPOTransformerModelStudent(PPOModel):
    def __init__(self,
                 new_arc: bool,
                 set_embed_dim: int,
                 way_embed_dim: int,
                 lock_embed_dim: int,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 hidden_dim_d: int,
                 window_size_s: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.new_arc = new_arc
        # self.set_embed_dim = set_embed_dim
        self.way_embed_dim = way_embed_dim
        self.window_size = window_size_s
        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.threshold = 0


        self.input_dim = (self.latency_dim + self.victim_acc_dim + self.way_embed_dim)

        # self.w_embed = nn.Embedding(self.way_embed_dim, self.way_embed_dim)
        # self.s_embed = nn.Embedding(self.set_embed_dim, self.set_embed_dim)

        # self.int_weighted_layer = nn.Linear(self.input_dim, hidden_dim_s, bias=False)
        # self.output_layer = nn.Linear(hidden_dim_s, 1)
        # self.linearxtra1 = nn.Linear(self.input_dim*self.window_size, 128, bias=False)
        # self.linearxtra2 = nn.Linear(128, self.input_dim*self.window_size, bias=False)
        self.linear = nn.Linear(self.input_dim*self.window_size, 1, bias=False)
        # self.linear0 = nn.Linear(self.input_dim*self.window_size, 128, bias=False)
        # self.linear1 = nn.Linear(128, 128, bias=False)
        # self.linear2 = nn.Linear(128, 128, bias=False)
        # self.linear3 = nn.Linear(128, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._device = None

    def make_one_hot(self, src: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = F.one_hot(src, num_classes)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def make_embedding(self, src: torch.Tensor,
                       embed: nn.Embedding) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = embed(src)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(torch.int64)
        # print("obs dimention: ", obs.shape, obs)
        if obs.dim() != 3:
            print(f"Assertion failed: obs.dim() is {obs.dim()}, expected 3")
            raise AssertionError(f"Expected obs.dim() == 3, but got {obs.dim()}")
        # batch_size = obs.size(0)
        # l, v, w, s = torch.unbind(obs, dim=-1)
        l, v, w = torch.unbind(obs, dim=-1)
        l = self.make_one_hot(l, self.latency_dim)
        v = self.make_one_hot(v, self.victim_acc_dim)
        w = self.make_one_hot(w, self.way_embed_dim)
        # s = self.make_embedding(s, self.s_embed)
        x = torch.cat((l, v, w), dim=-1)
        # x = torch.cat((l, v, w, s), dim=-1)
        x = x.view(x.size(0), -1)
        # for param in self.linear.parameters():
        #     param.data = torch.round(param.data)
        # print(f"x requires_grad before linear: {x.requires_grad}")
        # x = self.linear0(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        # output = self.linear3(x)
        x = x.float()
        output = self.linear(x)
        # print(f"output requires_grad after linear: {output.requires_grad}")
        # output.squeeze()
        # prediction = (output > self.threshold).int()
        prediction = self.sigmoid(output)
        # print(f"w_embed requires_grad: {self.w_embed.weight.requires_grad}")
        # print(f"s_embed requires_grad: {self.s_embed.weight.requires_grad}")
        # print(f"linear requires_grad: {self.linear.weight.requires_grad}")
        return prediction, output, x

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor, reload_model=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            #print(obs)
            x = obs.to(self._device)
            d = deterministic_policy.to(self._device)
            prediction, output = self.forward(x)
            
            # Ensure output is an integer (either 0 or 1)
            # output = (output > 0.5).int()
            # action = int(output.item())
            if output > 0.5:
                action = 1
            else:
                action = 0
            return action


class CachePPOTransformerModelPoolStudent(CachePPOTransformerModelStudent):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__(latency_dim,
                        victim_acc_dim,
                        action_dim,
                        step_dim,
                        window_size,
                        action_embed_dim,
                        step_embed_dim,
                        hidden_dim,
                        output_dim,
                        num_layers)
        self.history = []
        self.latest = None
        self.use_history = False

    @remote.remote_method(batch_size=128)
    def act(
        self, 
        obs: torch.Tensor, 
        deterministic_policy: torch.Tensor,
        reload_model: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reload_model:
            if self.use_history and len(self.history)>0:
                state_dict = random.choice(self.history)
                self.load_state_dict(state_dict)
            elif self.latest is not None:
                self.load_state_dict(self.latest)
            #print("reloading model", reload_model)
            #print("length of history:", len(self.history), "use history:", self.use_history, "latest:", self.latest if self.latest is None else len(self.latest))
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            try:
                x = obs.to(self._device)
            except: 
                print(obs)
            d = deterministic_policy.to(self._device)
            prediction, output = self.forward(x)
            
            # Ensure output is an integer (either 0 or 1)
            output = (output > 0.5).int()
            action = int(output.item())
            # if output > 0.5:
            #     action = 1
            # else:
            #     action = 0
            return action

    @remote.remote_method(batch_size=None)
    def push(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device), state_dict)
        self.latest = state_dict
        self.load_state_dict(state_dict)
    
    @remote.remote_method(batch_size=None)
    def push_to_history(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device), state_dict)
        self.latest = state_dict
        self.history.append(self.latest)
   
    @remote.remote_method(batch_size=None) 
    def set_use_history(self, use_history:bool) -> None:
        print("set use history", use_history)
        self.use_history = use_history
        print("after setting:", self.use_history)

class DownstreamModelPool(DownstreamModel):
    def __init__(self,
                 model: nn.Module,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 timeout: float = 60) -> None:
        super().__init__(model, server_name, server_addr, name, timeout)

    def set_use_history(self, use_history):
        self.client.sync(self.server_name, self.remote_method_name("set_use_history"),
                         use_history)
    
    def push_to_history(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name, self.remote_method_name("push_to_history"),
                         state_dict)    

ModelLike = Union[nn.Module, RemotableModel, DownstreamModel, remote.Remote]


def wrap_downstream_model(model: nn.Module,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModelPool(model, server.name, server.addr, name, timeout)
