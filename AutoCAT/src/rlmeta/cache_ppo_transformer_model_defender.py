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

class CachePPOTransformerModelDefender(PPOModel):
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
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.new_arc = new_arc
        self.set_embed_dim = set_embed_dim
        self.way_embed_dim = way_embed_dim
        self.lock_embed_dim = lock_embed_dim

        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        self.step_embed_dim = step_embed_dim
        # self.input_dim = (self.latency_dim + self.victim_acc_dim +
        #                   self.action_embed_dim + self.step_embed_dim)
        self.input_dim = (self.latency_dim + self.victim_acc_dim + self.way_embed_dim)
        self.hidden_dim = hidden_dim
        self.hidden_dim_d = hidden_dim_d
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        self.w_embed = nn.Embedding(self.way_embed_dim, self.way_embed_dim)
        self.v_embed = nn.Embedding(self.victim_acc_dim, self.victim_acc_dim)
        # self.s_embed = nn.Embedding(self.set_embed_dim, self.set_embed_dim)

        self.linear_i = nn.Linear(self.input_dim, self.hidden_dim_d)
        # self.linear_o = nn.Linear(self.hidden_dim * self.window_size,
        #                           self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim_d,
                                                   nhead=8,
                                                   dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.linear_v = nn.Linear(self.hidden_dim_d, 1)
        if self.new_arc:
            self.linear_o1 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o2 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o3 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o4 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o5 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o6 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o7 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o8 = nn.Linear(self.hidden_dim_d, 2)

            self.linear_o9 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o10 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o11 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o12 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o13 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o14 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o15 = nn.Linear(self.hidden_dim_d, 2)
            self.linear_o16 = nn.Linear(self.hidden_dim_d, 2)


        else:
            self.linear_a = nn.Linear(self.hidden_dim_d, self.output_dim)

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
        assert obs.dim() == 3

        # batch_size = obs.size(0)
        l, v, w = torch.unbind(obs, dim=-1)
        l = self.make_one_hot(l, self.latency_dim)
        v = self.make_embedding(v, self.v_embed)
        w = self.make_embedding(w, self.w_embed)

        # s = self.make_embedding(s, self.s_embed)
        
        x = torch.cat((l, v, w), dim=-1)
        # x = torch.zeros_like(x)
        x = self.linear_i(x)
        x = x.transpose(0, 1).contiguous()
        h = self.encoder(x)
        # h = self.linear_o(h.view(batch_size, -1))
        h = h.mean(dim=0)

        if self.new_arc:
            # Process layers 1 to 8
            p1 = self.linear_o1(h)
            logpi1 = F.log_softmax(p1, dim=-1)
            p2 = self.linear_o2(h)
            logpi2 = F.log_softmax(p2, dim=-1)
            p3 = self.linear_o3(h)
            logpi3 = F.log_softmax(p3, dim=-1)
            p4 = self.linear_o4(h)
            logpi4 = F.log_softmax(p4, dim=-1)
            p5 = self.linear_o5(h)
            logpi5 = F.log_softmax(p5, dim=-1)
            p6 = self.linear_o6(h)
            logpi6 = F.log_softmax(p6, dim=-1)
            p7 = self.linear_o7(h)
            logpi7 = F.log_softmax(p7, dim=-1)
            p8 = self.linear_o8(h)
            logpi8 = F.log_softmax(p8, dim=-1)

            p9 = self.linear_o9(h)
            logpi9 = F.log_softmax(p9, dim=-1)
            p10 = self.linear_o10(h)
            logpi10 = F.log_softmax(p10, dim=-1)
            p11 = self.linear_o11(h)
            logpi11 = F.log_softmax(p11, dim=-1)
            p12 = self.linear_o12(h)
            logpi12 = F.log_softmax(p12, dim=-1)
            p13 = self.linear_o13(h)
            logpi13 = F.log_softmax(p13, dim=-1)
            p14 = self.linear_o14(h)
            logpi14 = F.log_softmax(p14, dim=-1)
            p15 = self.linear_o15(h)
            logpi15 = F.log_softmax(p15, dim=-1)
            p16 = self.linear_o16(h)
            logpi16 = F.log_softmax(p16, dim=-1)

            # Concatenate the log probabilities from all 8 layers
            # logpi = torch.cat((logpi8, logpi7, logpi6, logpi5, logpi4, logpi3, logpi2, logpi1), dim=1)
            logpi = torch.cat((logpi16, logpi15, logpi14, logpi13, logpi12, logpi11, logpi10, logpi9, logpi8, logpi7, logpi6, logpi5, logpi4, logpi3, logpi2, logpi1), dim=1)

            # Reshape the result to match the new dimensions
            logpi = logpi.reshape(-1, 32)
        else:
            p = self.linear_a(h)
            logpi = F.log_softmax(p, dim=-1)

        v = self.linear_v(h)
        return logpi, v

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


class CachePPOTransformerModelPoolDefender(CachePPOTransformerModelDefender):
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
            logpi, v = self.forward(x)

            if self.new_arc:
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
                print("try to find the action*******************", greedy_action, "  ^^^^^^^^^^^^^", sample_action)
                action = torch.where(d, greedy_action, sample_action)
                logpi = logpi.gather(dim=-1, index=action)
                return action.cpu(), logpi.cpu(), v.cpu()

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
