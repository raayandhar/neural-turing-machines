# Addressing mechanisms for NTM

import torch
import torch.nn as nn
import torch.nn.functional as F


class Addressing(nn.Module):
    def __init__(self, memory_size, controller_size):

        super(Addressing, self).__init__()
        self.memory_size = memory_size

        self.key_fc = nn.Linear(controller_size, memory_size)  # For computing the key for content-based addressing
        self.beta_fc = nn.Linear(controller_size, 1)  # For computing the beta parameter for sharpening the weights

        self.gate_fc = nn.Linear(controller_size, 1)  # For computing the gate for content-based addressing
        self.shift_fc = nn.Linear(controller_size, 3) # Shift weights for the addressing mechanism (left, right, or no shift)
        self.gamma_fc = nn.Linear(controller_size, 1)  # Sharpening factor for the addressing weights

    def content_addressing(self, memory, k, beta):
        """
        Content-based addressing mechanism.
        
        Args:
            memory (torch.Tensor): The memory from which to read, shape (batch_size, N, M).
            k (torch.Tensor): The key for content-based addressing, shape (batch_size, N).
            beta (torch.Tensor): Sharpening parameter, shape (batch_size, 1).
        
        Returns:
            torch.Tensor: Addressing weights, shape (batch_size, N).
        """
        k = k.unsqueeze(-1)
        memory_norm = F.normalize(memory, p=2, dim=2)
        k_norm = F.normalize(k, p=2, dim=2)

        similarity = torch.matmul(memory_norm, k_norm.transpose(1,2)).squeeze(2)
        weighted_similarity = similarity * beta.squeeze(1)

        content_w = F.softmax(weighted_similarity, dim=1)

        return content_w

    def location_addressing(self, prev_w, g, s, gamma):
        """
        Location-based addressing mechanism.
        
        Args:
            prev_w (torch.Tensor): Previous addressing weights, shape (batch_size, N).
            g (torch.Tensor): Gate for location-based addressing, shape (batch_size, 1).
            s (torch.Tensor): Shift weights, shape (batch_size, 3).
            gamma (torch.Tensor): Sharpening factor for the addressing weights, shape (batch_size, 1).
        
        Returns:
            torch.Tensor: Addressing weights after location-based addressing, shape (batch_size, N).
        """
        batch_size, N = prev_w.size()
        
        g = torch.sigmoid(g).squeeze(1)
        w_g = g.unsqueeze(1) * prev_w  # Apply the gate to the previous weights

        s_norm = F.softmax(s, dim=1)

        w_shifted = torch.zeros_like(w_g)
        for b in range(batch_size):
            w_shifted[b] = torch.roll(w_g[b], -1) * s_norm[b, 0] + \
                          w_g[b] * s_norm[b, 1] + \
                          torch.roll(w_g[b], 1) * s_norm[b, 2]

        # Apply sharpening
        gamma = 1 + F.softplus(gamma).squeeze(1)
        w_sharp = w_shifted ** gamma.unsqueeze(1)
        w_sharp = w_sharp / (w_sharp.sum(dim=1, keepdim=True) + 1e-6)

        return w_sharp
        
        
        
        
