import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class PolicyModel(nn.Module):
    def __init__(self, view_h, view_w, num_states=4, num_action=4, hidden_size=32):
        super().__init__()
        self.view_h = view_h
        self.view_w = view_w
        self.num_states = num_states
        self.num_action = num_action
        self.hidden_size = hidden_size
        self.conv1 = conv3x3(num_states, hidden_size)
        self.linear1 = nn.Linear(hidden_size * view_h * view_w, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_action)
    
    def forward(self, x):
        # b, h, w (long)
        x = F.one_hot(x, num_classes=self.num_states).permute(0,3,1,2).contiguous().float()
        x = self.conv1(x).relu()
        x = x.view(-1, self.hidden_size * self.view_h * self.view_w)
        x = self.linear1(x).relu()
        x = self.linear2(x)
        x = x.softmax(dim=-1)
        return x


def decode_action(x):
    decode_mapping = {
        0: 'left',
        1: 'right',
        2: 'up',
        3: 'down',
    }
    action = []
    for b in range(x.shape[0]):
        action.append(decode_mapping[x[b].item()])
    return action
