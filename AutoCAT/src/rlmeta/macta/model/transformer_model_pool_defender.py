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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from cache_ppo_transformer_model_defender import CachePPOTransformerModelDefender


class CachePPOTransformerModelPoolDefender(CachePPOTransformerModelDefender):
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
        super().__init__(new_arc, set_embed_dim, way_embed_dim, lock_embed_dim, latency_dim, victim_acc_dim, action_dim, step_dim,
                         window_size, action_embed_dim, step_embed_dim,
                         hidden_dim, hidden_dim_d, output_dim, num_layers)
        self.history = []
        self.latest = None
        self.use_history = False
        self.new_arc = new_arc

    # @remote.remote_method(batch_size=128)
    # def act(self, obs: torch.Tensor, deterministic_policy: torch.Tensor,
    #         reload_model: bool
    #         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if reload_model:
    #         if self.use_history and len(self.history) > 0:
    #             state_dict = random.choice(self.history)
    #             self.load_state_dict(state_dict)
    #         elif self.latest is not None:
    #             self.load_state_dict(self.latest)
    #         #print("reloading model", reload_model)
    #         #print("length of history:", len(self.history), "use history:", self.use_history, "latest:", self.latest if self.latest is None else len(self.latest))

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.use_history and len(self.history) > 0:
            state_dict = random.choice(self.history)
            self.load_state_dict(state_dict)
        elif self.latest is not None:
            self.load_state_dict(self.latest)

        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            try:
                x = obs.to(self._device)
            except:
                print("look at obs --------------------------------------------------> ", obs)
            d = deterministic_policy.to(self._device)
            logpi, v = self.forward(x)

            if self.new_arc:
                # Split logpi into 16 layers
                logpi_layer1 = logpi[:, :2]    # Log probabilities for output layer 1
                logpi_layer2 = logpi[:, 2:4]   # Log probabilities for output layer 2
                logpi_layer3 = logpi[:, 4:6]   # Log probabilities for output layer 3
                logpi_layer4 = logpi[:, 6:8]   # Log probabilities for output layer 4
                logpi_layer5 = logpi[:, 8:10]  # Log probabilities for output layer 5
                logpi_layer6 = logpi[:, 10:12] # Log probabilities for output layer 6
                logpi_layer7 = logpi[:, 12:14] # Log probabilities for output layer 7
                logpi_layer8 = logpi[:, 14:16]   # Log probabilities for output layer 8

                logpi_layer9 = logpi[:, 16:18]    # Log probabilities for output layer 9
                logpi_layer10 = logpi[:, 18:20]   # Log probabilities for output layer 10
                logpi_layer11 = logpi[:, 20:22]   # Log probabilities for output layer 11
                logpi_layer12 = logpi[:, 22:24]   # Log probabilities for output layer 12
                logpi_layer13 = logpi[:, 24:26]  # Log probabilities for output layer 13
                logpi_layer14 = logpi[:, 26:28] # Log probabilities for output layer 14
                logpi_layer15 = logpi[:, 28:30] # Log probabilities for output layer 15
                logpi_layer16 = logpi[:, 30:]   # Log probabilities for output layer 16


                # Sample actions
                
                sample_action1 = logpi_layer1.exp().multinomial(1, replacement=True)
                sample_action2 = logpi_layer2.exp().multinomial(1, replacement=True)
                sample_action3 = logpi_layer3.exp().multinomial(1, replacement=True)
                sample_action4 = logpi_layer4.exp().multinomial(1, replacement=True)
                sample_action5 = logpi_layer5.exp().multinomial(1, replacement=True)
                sample_action6 = logpi_layer6.exp().multinomial(1, replacement=True)
                sample_action7 = logpi_layer7.exp().multinomial(1, replacement=True)
                sample_action8 = logpi_layer8.exp().multinomial(1, replacement=True)

                sample_action9 = logpi_layer9.exp().multinomial(1, replacement=True)
                sample_action10 = logpi_layer10.exp().multinomial(1, replacement=True)
                sample_action11 = logpi_layer11.exp().multinomial(1, replacement=True)
                sample_action12 = logpi_layer12.exp().multinomial(1, replacement=True)
                sample_action13 = logpi_layer13.exp().multinomial(1, replacement=True)
                sample_action14 = logpi_layer14.exp().multinomial(1, replacement=True)
                sample_action15 = logpi_layer15.exp().multinomial(1, replacement=True)
                sample_action16 = logpi_layer16.exp().multinomial(1, replacement=True)


                # Greedy actions
                greedy_action1 = logpi_layer1.argmax(-1, keepdim=True)
                greedy_action2 = logpi_layer2.argmax(-1, keepdim=True)
                greedy_action3 = logpi_layer3.argmax(-1, keepdim=True)
                greedy_action4 = logpi_layer4.argmax(-1, keepdim=True)
                greedy_action5 = logpi_layer5.argmax(-1, keepdim=True)
                greedy_action6 = logpi_layer6.argmax(-1, keepdim=True)
                greedy_action7 = logpi_layer7.argmax(-1, keepdim=True)
                greedy_action8 = logpi_layer8.argmax(-1, keepdim=True)

                greedy_action9 = logpi_layer9.argmax(-1, keepdim=True)
                greedy_action10 = logpi_layer10.argmax(-1, keepdim=True)
                greedy_action11 = logpi_layer11.argmax(-1, keepdim=True)
                greedy_action12 = logpi_layer12.argmax(-1, keepdim=True)
                greedy_action13 = logpi_layer13.argmax(-1, keepdim=True)
                greedy_action14 = logpi_layer14.argmax(-1, keepdim=True)
                greedy_action15 = logpi_layer15.argmax(-1, keepdim=True)
                greedy_action16 = logpi_layer16.argmax(-1, keepdim=True)

                # Select actions based on the condition `d`
                action1 = torch.where(d, greedy_action1, sample_action1)
                action2 = torch.where(d, greedy_action2, sample_action2)
                action3 = torch.where(d, greedy_action3, sample_action3)
                action4 = torch.where(d, greedy_action4, sample_action4)
                action5 = torch.where(d, greedy_action5, sample_action5)
                action6 = torch.where(d, greedy_action6, sample_action6)
                action7 = torch.where(d, greedy_action7, sample_action7)
                action8 = torch.where(d, greedy_action8, sample_action8)

                action9 = torch.where(d, greedy_action9, sample_action9)
                action10 = torch.where(d, greedy_action10, sample_action10)
                action11 = torch.where(d, greedy_action11, sample_action11)
                action12 = torch.where(d, greedy_action12, sample_action12)
                action13 = torch.where(d, greedy_action13, sample_action13)
                action14 = torch.where(d, greedy_action14, sample_action14)
                action15 = torch.where(d, greedy_action15, sample_action15)
                action16 = torch.where(d, greedy_action16, sample_action16)


                # Combine actions
                # combined_action = torch.cat((action1, action2, action3, action4, action5, action6, action7, action8), dim=1)
                combined_action = torch.cat((action1, action2, action3, action4, action5, action6, action7, action8, action9, action10, action11, action12, action13, action14, action15, action16), dim=1) 
                new_action = torch.sum(combined_action * (2 ** torch.arange(combined_action.size(1)-1, -1, -1).float().to(combined_action.device)), dim=1)


                # Gather the log probabilities corresponding to the selected actions
                logpi1 = logpi_layer1.gather(dim=-1, index=action1)
                logpi2 = logpi_layer2.gather(dim=-1, index=action2)
                logpi3 = logpi_layer3.gather(dim=-1, index=action3)
                logpi4 = logpi_layer4.gather(dim=-1, index=action4)
                logpi5 = logpi_layer5.gather(dim=-1, index=action5)
                logpi6 = logpi_layer6.gather(dim=-1, index=action6)
                logpi7 = logpi_layer7.gather(dim=-1, index=action7)
                logpi8 = logpi_layer8.gather(dim=-1, index=action8)

                logpi9 = logpi_layer9.gather(dim=-1, index=action9)
                logpi10 = logpi_layer10.gather(dim=-1, index=action10)
                logpi11 = logpi_layer11.gather(dim=-1, index=action11)
                logpi12 = logpi_layer12.gather(dim=-1, index=action12)
                logpi13 = logpi_layer13.gather(dim=-1, index=action13)
                logpi14 = logpi_layer14.gather(dim=-1, index=action14)
                logpi15 = logpi_layer15.gather(dim=-1, index=action15)
                logpi16 = logpi_layer16.gather(dim=-1, index=action16)


                # Sum the log probabilities for all actions
                # total_logpi = logpi1 + logpi2 + logpi3 + logpi4 + logpi5 + logpi6 + logpi7 + logpi8 
                total_logpi = logpi1 + logpi2 + logpi3 + logpi4 + logpi5 + logpi6 + logpi7 + logpi8 + logpi9 + logpi10 + logpi11 + logpi12 + logpi13 + logpi14 + logpi15 + logpi16
                return new_action.cpu(), total_logpi.cpu(), v.cpu()

            else:
                greedy_action = logpi.argmax(-1, keepdim=True)
                sample_action = logpi.exp().multinomial(1, replacement=True)
                action = torch.where(d, greedy_action, sample_action)
                logpi = logpi.gather(dim=-1, index=action)

                return action.cpu(), logpi.cpu(), v.cpu()
            

            '''
            new_action = []
            total_logpi = []
            if len(logpi.shape) == 1:
                max_logpi = 0
                reshaped_tensor = logpi.reshape(-1, 2)
                max_indices = torch.argmax(reshaped_tensor, dim=1)
                max_logpi = torch.max(reshaped_tensor, dim=1)[0].sum().item()
                actual_action = 0
                for i in range(len(max_indices)):
                    actual_action += int(max_indices[i]) * (2**i)
                new_action = torch.tensor(actual_action)
                total_logpi = torch.tensor(max_logpi)
            else:
                for lgp in logpi:
                    max_logpi = 0
                    reshaped_tensor = lgp.reshape(-1, 2)
                    max_indices = torch.argmax(reshaped_tensor, dim=1)
                    max_logpi = torch.max(reshaped_tensor, dim=1)[0].sum().item()
                    total_logpi.append(max_logpi)
                    actual_action = 0
                    for i in range(len(max_indices)):
                        actual_action += int(max_indices[i]) * (2**i)
                    new_action.append(actual_action)
                new_action = torch.tensor(new_action)
                total_logpi = torch.tensor(total_logpi)
            tensor_list = [torch.tensor([value]) for value in new_action]
            new_action = torch.stack(tensor_list)
            tensor_list = [torch.tensor([value]) for value in total_logpi]
            total_logpi = torch.stack(tensor_list)
            '''
            # print("this is in pool defender   ", total_logpi)

            # print(new_action, "  __---__--__--  ", total_logpi)

            # print("^^^^^^^^^^^in cache ppo transformer model defender  ", type(deterministic_policy), "  _______________________", deterministic_policy)
            # print("*******", logpi)
            ##greedy_action = logpi.argmax(-1, keepdim=True)
            ########logpi = logpi.gather(dim=-1, index=greedy_action)
            ########indices = torch.eq(greedy_action, 1)
            ########greedy_action[indices] = 15
            ##sample_action = logpi.exp().multinomial(1, replacement=True)
            ##action = torch.where(d, greedy_action, sample_action)
            # print(action, "  ()()()()() ", type(action))
            # print("^^^^^^^^^^^in cache ppo transformer model defender  ", sample_action, sample_action.shape, type(sample_action), "  _______________________", logpi)
            # print("^^^^^^^^^^^in cache ppo transformer model defender  ", action.shape, type(int(logpi[0][3])), int(logpi[0][3]), sample_action[0][1], "  _______________________", logpi)
            # print("$$$$$$$$", torch.exp(logpi))
            ##logpi = logpi.gather(dim=-1, index=action)
            # print("*********", logpi)
            # print("&&&&&&&&&", total_logpi, type(total_logpi))
            # print("**************************in cache ppo transformer model defender  ", greedy_action, "%%%%%%%%%%%%", sample_action, "  &&&&&&&&&&&&&&&&&&&&&&&&", logpi)
            ####greedy_action = logpi.argmax(-1, keepdim=True)
            ####sample_action = logpi.exp().multinomial(1, replacement=True)
            ####action = torch.where(d, greedy_action, sample_action)
            ####logpi = logpi.gather(dim=-1, index=action)
            ##return action.cpu(), logpi.cpu(), v.cpu()
            # return new_action.cpu(), total_logpi.cpu(), v.cpu()
            ########return greedy_action.cpu(), logpi.cpu(), v.cpu()

    @remote.remote_method(batch_size=None)
    def push(self, state_dict: Dict[str, torch.Tensor]) -> None:
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device),
                                             state_dict)
        self.latest = state_dict
        self.load_state_dict(state_dict)

    @remote.remote_method(batch_size=None)
    def push_to_history(self, state_dict: Dict[str, torch.Tensor]) -> None:
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device),
                                             state_dict)
        self.latest = state_dict
        self.history.append(self.latest)

    @remote.remote_method(batch_size=None)
    def set_use_history(self, use_history: bool) -> None:
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
        self.client.sync(self.server_name,
                         self.remote_method_name("set_use_history"),
                         use_history)

    def push_to_history(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name,
                         self.remote_method_name("push_to_history"),
                         state_dict)


ModelLike = Union[nn.Module, RemotableModel, DownstreamModel, remote.Remote]


def wrap_downstream_model(model: nn.Module,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModelPool(model, server.name, server.addr, name, timeout)
