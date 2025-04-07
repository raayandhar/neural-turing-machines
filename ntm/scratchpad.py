# "scratchpad" or memory attached to our model.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Scratchpad(nn.Module):
    def __init__(self, N, M):
        """
        Args:
                N (int): Number of rows in the scratchpad.
                M (int): Number of columns in the scratchpad.       
        """
        super(Scratchpad, self).__init__()
        self.N = N
        self.M = M
        self.register_buffer('memory', torch.zeros(N, M))  # Persistent buffer for the memory.

    def reset(self, batch_size):
        self.memory = torch.zeros(batch_size, self.N, self.M).to(self.memory.device)  # Reset the memory to zeros.

    def read(self, w):
        """
        Read from the scratchpad using weights `w`.
        
        Args:
            w (torch.Tensor): Weights for reading, shape (batch_size, N).
        
        Returns:
            torch.Tensor: The read values from the scratchpad, shape (batch_size, M).
        """
        return torch.matmul(self.memory, w.unsqueeze(-1)).squeeze(-1)

    def write(self, w, e, a):
        """
        Write to the scratchpad using weights `w`, erase vector `e`, and add vector `a`.
        
        Args:
            w (torch.Tensor): Weights for writing, shape (batch_size, N).
            e (torch.Tensor): Erase vector, shape (batch_size, M).
            a (torch.Tensor): Add vector, shape (batch_size, M).
        
        Returns:
            None: Updates the memory in place.
        """
        # Compute the contribution to the memory from the write weights.
        contribution = w.unsqueeze(-1) * (1 - e.unsqueeze(-1)) + a.unsqueeze(-1)
        self.memory += contribution
        
        
