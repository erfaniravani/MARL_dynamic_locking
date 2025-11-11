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

from cache_ppo_transformer_model_student import CachePPOTransformerModelStudent


class CachePPOTransformerModelPoolStudent(CachePPOTransformerModelStudent):
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
                         hidden_dim, hidden_dim_d, window_size_s, output_dim, num_layers)
        self.history = []
        self.latest = None
        self.use_history = False
        self.new_arc = new_arc
        self.learning_rate=0.0001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
    def update_weights(self, student_action, teacher_action):


        for param in self.parameters():
            param.requires_grad = True

        # teacher_action.requires_grad = False
        # student_action.requires_grad = True

        # print(f"student_action: {type(student_action)}, teacher_action: {type(teacher_action)}")
        # print(f"student_action.requires_grad: {student_action.requires_grad}")
        # print(f"teacher_action.requires_grad: {teacher_action.requires_grad}")

        # Ensure shapes match
        # print(f"student_action shape: {student_action.shape}")
        # print(f"teacher_action shape: {teacher_action.shape}") 
        assert student_action.shape == teacher_action.shape, "Shapes of student_action and teacher_action must match"
        
        # # Calculate the update direction and magnitude
        # update = (teacher_action - student_action).float()  # Shape: (batch_size,)

        # # Scale the updates by the learning rate
        # scaled_update = self.learning_rate * update  # Shape: (batch_size,)

        # # Apply updates to weights
        # for param in self.linear.parameters():
        #     # Adjust shape of scaled_update to match param.data
        #     if param.data.ndimension() == 2:  # For 2D weights
        #         # print("weight update dimension 2", scaled_update, scaled_update.shape)
        #         param.data += scaled_update.view(-1, 1).sum(dim=0)
        #     elif param.data.ndimension() == 1:  # For 1D biases or weights
        #         print("weight update dimension 1")
        #         param.data += scaled_update.sum()
        
        
        criterion = nn.BCELoss()  # Define the loss function
        loss = criterion(student_action, teacher_action)
        isvm_loss = torch.abs((teacher_action - student_action).float())
        isvm_loss = isvm_loss.view(-1, 1).sum(dim=0)
        criterion1 = nn.MSELoss()
        l2_loss = criterion1(student_action, teacher_action)
        criterion2 = nn.CrossEntropyLoss()
        ent_loss = criterion2(student_action, teacher_action)
        # print("this is loss: ", loss, type(loss), loss.shape)
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if "linear" in name:  # Filter for the linear layer
        #         if param.grad is None:
        #             print(f"{name} grad is None")
        #         else:
        #             print(f"{name} grad norm: {param.grad.norm()}") #param.grad.norm().mean().item()
        self.optimizer.step()
        param = self.linear.weight
        return loss.mean().item(), param.grad.norm().mean().item(), isvm_loss, l2_loss, ent_loss

    @remote.remote_method(batch_size=128)
    def get_loss(self, student_action, teacher_action, learning_rate=0.0001):
        # Ensure shapes match
        assert student_action.shape == teacher_action.shape, "Shapes of student_action and teacher_action must match"
        
        # Calculate the update direction and magnitude
        update = (teacher_action - student_action).float()  # Shape: (batch_size,)
        # print('update: ', update.view(-1, 1).sum(dim=0))
        return update.view(-1, 1).sum(dim=0)
        


    @remote.remote_method(batch_size=128)
    def f_pass(self, obs: torch.Tensor, deterministic_policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_history and len(self.history) > 0:
            state_dict = random.choice(self.history)
            self.load_state_dict(state_dict)
        elif self.latest is not None:
            self.load_state_dict(self.latest)

        if self._device is None:
            self._device = next(self.parameters()).device

        try:
            x = obs.to(self._device)
        except:
            print("look at obs --------------------------------------------------> ", obs)
        d = deterministic_policy.to(self._device)
        if self._device is None:
            self._device = next(self.parameters()).device

        #print(obs)
        x = obs.to(self._device)
        prediction, output, x = self.forward(x)
        # print("from f_pass: ", prediction)
        return prediction
    
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

       
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            try:
                x = obs.to(self._device)
            except:
                print("look at obs --------------------------------------------------> ", obs)
            d = deterministic_policy.to(self._device)
            x = x.unsqueeze(0)
            prediction, output, x = self.forward(x)
            
            # # Ensure output is an integer (either 0 or 1)
            # output = (output > 0).int()
            # action = int(output.item())
            # if output >= 0.5:
            #     action = 1
            # else:
            #     action = 0
            prediction = (prediction >= 0.5)
            action = prediction.int()
            return action, 1, 2
            


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
