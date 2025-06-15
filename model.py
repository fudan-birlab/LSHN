import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

from typing import Dict

def attractor_activation(v): # v -> hidden_state
    return v.clamp(-1., 1.)

def attractor_v_clamp(v):
    return v.clamp(-1., 1.)

class attractorRNNCell(nn.RNNCell):

    def set_dt(self, dt):
        self.dt = dt

    def attractorRnn_cell(self, input, v, w_ih, w_hh, b_ih, b_hh, enable_detach=False, detach_recurrent=False):
        v = attractor_v_clamp(v)
        if enable_detach:
            v = v.detach()
        hidden = attractor_activation(v)
        igates = torch.mm(input, w_ih.t()) + b_ih
        if detach_recurrent:
            hgates = torch.mm(hidden, w_hh.t().data.detach()) + b_hh
        else:
            hgates = torch.mm(hidden, w_hh.t()) + b_hh
        v_next = attractor_v_clamp(self.dt * (igates + hgates) + v)
        hidden_next = attractor_activation(v_next)
        return v_next, hidden_next

    def forward(self, input: torch.Tensor, v, enable_detach=False, detach_recurrent=False) -> torch.Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"attractorRNNCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if v is not None and v.dim() not in (1, 2):
            raise ValueError(
                f"attractorRNNCell: Expected voltage to be 1D or 2D, got {v.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if v is None:
            v = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            v = v.unsqueeze(0) if not is_batched else v

        v_next, hv_next = self.attractorRnn_cell(
            input,
            v,
            self.weight_ih,
            self.weight_hh,
            0 if self.bias_ih is None else self.bias_ih,
            0 if self.bias_hh is None else self.bias_hh,
            enable_detach=enable_detach,
            detach_recurrent=detach_recurrent
        )

        if not is_batched:
            v_next = v_next.squeeze(0)
            hv_next = hv_next.squeeze(0)

        return v_next, hv_next

class EpisodicMemory(nn.Module):

    def __init__(self, attribute_list, ento_size, em_size, event_size, drop, additional_iteration=0):
        super().__init__()

        self.ento_size = ento_size
        self.em_size = em_size
        self.event_size = event_size

        self.attribute_list = attribute_list

        self.drop = drop
        self.input_dropout = nn.Dropout(drop)

        self.attractor_RNN = nn.ModuleDict({
            attribute : attractorRNNCell(input_size=self.ento_size+(len(self.attribute_list)-1)*self.em_size, hidden_size=self.em_size, bias=False, nonlinearity='tanh')
            for attribute in self.attribute_list
        })
        self.set_dt(1.)

        self.additional_iteration = additional_iteration

    def set_dt(self, dt):
        for attribute in self.attribute_list:
            self.attractor_RNN[attribute].set_dt(dt)

    def reset(self, batch_size, device):
        self.neuron_v = {
            neuron : torch.rand(batch_size, self.em_size, device=device) * 2. - 1.
            for neuron in self.attribute_list
        }
        self.neuron_h = {
            neuron : attractor_activation(self.neuron_v[neuron])
            for neuron in self.attribute_list
        }

    def step_forward(self, ento_feat, return_dict=True):
        B, N = ento_feat.shape
        ento_feat = self.input_dropout(ento_feat)
        assert N == self.ento_size
        neuron_input = {
            neuron : torch.cat([ento_feat] + [self.neuron_v[pre] for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        for neuron in self.attribute_list:
            self.neuron_v[neuron], self.neuron_h[neuron] = self.attractor_RNN[neuron](neuron_input[neuron], self.neuron_v[neuron], enable_detach=True)

        for _ in range(self.additional_iteration):
            self.neuron_v, self.neuron_h = self.attractor_iteration(self.neuron_v, self.neuron_h)

        if return_dict:
            return {neuron : self.neuron_h[neuron].clone() for neuron in self.attribute_list}
        else:
            return torch.cat([self.neuron_h[neuron] for neuron in self.attribute_list], -1)

    def forward(self, ento_feat):
        T, B, N = ento_feat.shape
        ento_feat = self.input_dropout(ento_feat)
        assert N == self.ento_size
        neuron_v = {
            neuron : torch.rand(B, self.hidden_size).to(ento_feat.device) * 2. - 1.
            for neuron in self.attribute_list
        }
        neuron_h = {
            neuron : attractor_activation(neuron_v[neuron])
            for neuron in self.attribute_list
        }
        outputs = {
            neuron : []
            for neuron in self.attribute_list
        }
        for i in range(T):
            neuron_input = {
                neuron : torch.cat([ento_feat[i]] + [neuron_v[pre] for pre in self.attribute_list if pre != neuron], -1)
                for neuron in self.attribute_list
            }
            for neuron in self.attribute_list:
                neuron_v[neuron], neuron_h[neuron] = self.attractor_RNN[neuron](neuron_input[neuron], neuron_v[neuron])
            for _ in range(self.additional_iteration):
                neuron_v, neuron_h = self.attractor_iteration(neuron_v, neuron_h)
            for neuron in self.attribute_list:
                outputs[neuron].append(neuron_h[neuron][None])
            
        return {neuron : torch.cat(outputs[neuron], 0) for neuron in self.attribute_list}

    def inter_conn(self, neuron1, neuron2):
        assert neuron1 != neuron2
        idx1 = np.where(np.array(self.attribute_list) == neuron1)[0].item()
        idx2 = np.where(np.array(self.attribute_list) == neuron2)[0].item()
        if idx1 < idx2:
            idx2 -= 1
        else:
            idx1 -= 1
        return idx1, idx2, (self.attractor_RNN[neuron1].weight_ih[:,self.ento_size+idx2*self.em_size:self.ento_size+(idx2+1)*self.em_size] + self.attractor_RNN[neuron2].weight_ih[:,self.ento_size+idx1*self.em_size:self.ento_size+(idx1+1)*self.em_size].transpose(1, 0)) / 2.

    @torch.no_grad()
    def constrain_inter_conn(self, neuron1, neuron2):
        idx1, idx2, new_conn = self.inter_conn(neuron1, neuron2)
        new_conn = new_conn.detach()
        self.attractor_RNN[neuron1].weight_ih.data[:,self.ento_size+idx2*self.em_size:self.ento_size+(idx2+1)*self.em_size] = new_conn
        self.attractor_RNN[neuron2].weight_ih.data[:,self.ento_size+idx1*self.em_size:self.ento_size+(idx1+1)*self.em_size] = new_conn

    @torch.no_grad()
    def constrain_attractor(self):
        for i, neuron1 in enumerate(self.attribute_list):
            self.attractor_RNN[neuron1].weight_hh.data = (self.attractor_RNN[neuron1].weight_hh.data + self.attractor_RNN[neuron1].weight_hh.data.transpose(1, 0)) / 2.
            for neuron2 in self.attribute_list[i+1:]:
                self.constrain_inter_conn(neuron1, neuron2)

    def Tensor2dict(self, x):
        x = torch.chunk(x, chunks=len(self.attribute_list), dim=-1)
        return {self.attribute_list[i]:x[i] for i in range(len(self.attribute_list))}

    def eventDict2Tensor(self, event:Dict):
        return torch.cat([event[attribute] for attribute in self.attribute_list], -1)

    def attractor_iteration(self, target_v:Dict, target_h:Dict):
        origin_shape = target_v[self.attribute_list[0]].shape
        target_v = {k:target_v[k].reshape(-1, self.em_size) for k in target_v.keys()}
        target_h = {k:target_h[k].reshape(-1, self.em_size) for k in target_h.keys()}
        neuron_input_withconn = {
            neuron : torch.cat([torch.zeros(target_h[neuron].shape[0], self.ento_size, device=target_h[neuron].device)] + [target_h[pre] for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        for neuron in self.attribute_list:
            target_v[neuron], target_h[neuron] = self.attractor_RNN[neuron](neuron_input_withconn[neuron], target_v[neuron])
            target_v[neuron] = target_v[neuron].reshape(*origin_shape)
            target_h[neuron] = target_h[neuron].reshape(*origin_shape)
        return target_v, target_h

    def set_inter_conn(self, neuron1, neuron2, w):
        assert neuron1 != neuron2
        idx1 = np.where(np.array(self.attribute_list) == neuron1)[0].item()
        idx2 = np.where(np.array(self.attribute_list) == neuron2)[0].item()
        if idx1 < idx2:
            idx2 -= 1
        else:
            idx1 -= 1
        self.attractor_RNN[neuron1].weight_ih.data[:,self.ento_size+idx2*self.em_size:self.ento_size+(idx2+1)*self.em_size] = w

    def set_whole_RNN_weight(self, W):
        for i, neuron1 in enumerate(self.attribute_list):
            for j, neuron2 in enumerate(self.attribute_list):
                w = W[i*self.em_size:(i+1)*self.em_size,j*self.em_size:(j+1)*self.em_size]
                if neuron1 == neuron2:
                    self.attractor_RNN[neuron1].weight_hh.data = w
                else:
                    self.set_inter_conn(neuron1, neuron2, w)

    def whole_RNN_weight(self):
        w_list = []
        for neuron1 in self.attribute_list:
            for neuron2 in self.attribute_list:
                if neuron1 == neuron2:
                    w_list.append(self.attractor_RNN[neuron1].weight_hh)
                else:
                    w_list.append(self.inter_conn(neuron1, neuron2)[-1])
        W = torch.cat(w_list, -1)
        return torch.cat(W.chunk(len(self.attribute_list), dim=-1), 0)

    def attractor_loss(self, target:Dict, noise_scale=0., mask_attr=[]):
        target = {k:target[k].reshape(-1, self.em_size) for k in target.keys()}
        loss = 0.

        # pepper noise
        state_next_withconn    = {neuron:target[neuron] * (1.0 * (torch.rand(target[neuron].shape, device=target[neuron].device) > noise_scale)) for neuron in self.attribute_list}
        state_next_withoutconn = {neuron:target[neuron] * (1.0 * (torch.rand(target[neuron].shape, device=target[neuron].device) > noise_scale)) for neuron in self.attribute_list}
        for attr in mask_attr:
            state_next_withconn[attr] *= 0.
        state_next_withconn_v    = {neuron:1*state_next_withconn[neuron] for neuron in self.attribute_list}
        state_next_withoutconn_v = {neuron:1*state_next_withoutconn[neuron] for neuron in self.attribute_list}

        neuron_input_withconn = {
            neuron : torch.cat([torch.zeros(target[neuron].shape[0], self.ento_size, device=target[neuron].device)] + [state_next_withconn[pre].clone() for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }
        neuron_input_withoutconn = {
            neuron : torch.cat([torch.zeros(target[neuron].shape[0], self.ento_size, device=target[neuron].device)] + [torch.zeros_like(state_next_withconn[pre], device=state_next_withconn[pre].device) for pre in self.attribute_list if pre != neuron], -1)
            for neuron in self.attribute_list
        }

        for _ in range(self.additional_iteration + 1):
            for neuron in self.attribute_list:
                state_next_withconn_v[neuron], state_next_withconn[neuron] = self.attractor_RNN[neuron](neuron_input_withconn[neuron], state_next_withconn_v[neuron], enable_detach=True)
                loss = loss + F.mse_loss(state_next_withconn[neuron], target[neuron])

        return loss / len(self.attribute_list) / (self.additional_iteration + 1)

class LSHN(EpisodicMemory):
    
    def __init__(self, attribute_list, em_size, image_size):
        super().__init__(
            attribute_list=attribute_list,
            ento_size=em_size,
            em_size=em_size,
            event_size=1,
            drop=0.,
            additional_iteration=10,
        )
        self.set_whole_RNN_weight(self.whole_RNN_weight().detach() * 0.)

        self.image_size = image_size
        
        self.image_to_attractor = nn.Sequential(
            nn.Linear(image_size, int((image_size*em_size)**0.5), bias=True),
            nn.GELU(),
            nn.Linear(int((image_size*em_size)**0.5), em_size, bias=True),
            nn.Tanh(),
        )
        self.attractor_to_image = nn.Sequential(
            nn.Linear(em_size, int((image_size*em_size)**0.5), bias=True),
            nn.GELU(),
            nn.Linear(int((image_size*em_size)**0.5), image_size, bias=True),
            nn.Tanh(),
        )

        for m in [self.image_to_attractor[0], self.attractor_to_image[0], self.image_to_attractor[2], self.attractor_to_image[2]]:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
        
    def attractor_loss(self, state, target):
        loss = 0.
        neuron = self.attribute_list[0]

        state_next_withconn    = state.clone()
        state_next_withconn_v  = state_next_withconn.clone()

        neuron_input_withconn = state.clone()

        for _ in range(self.additional_iteration + 1):
            state_next_withconn_v, state_next_withconn = self.attractor_RNN[neuron](neuron_input_withconn, state_next_withconn_v, enable_detach=True)
            loss = loss + F.mse_loss(state_next_withconn, target)
            loss = loss + F.l1_loss(state_next_withconn, target)

        return loss / len(self.attribute_list) / (self.additional_iteration + 1)

    def attractor_iteration(self, state, ento_input=None):
        neuron = self.attribute_list[0]
        if ento_input is None:
            neuron_input_withconn = torch.zeros(state.shape[0], self.ento_size, device=state.device)
        else:
            neuron_input_withconn = ento_input.clone()
        state_next_v, state_next = self.attractor_RNN[neuron](neuron_input_withconn, state.clone(), enable_detach=True)
        
        return state_next

    def save_model(self, path):
        torch.save(self.state_dict(), path)
