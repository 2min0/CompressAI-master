import torch
import torch.nn as nn

x = torch.tensor([[1,2,1,2], [3,4, 3,4],[1,2,1,2], [3,4,33,44]]).view((1,1,4,4))
print(x),
pixelshuffle = nn.PixelUnshuffle(2)
y = pixelshuffle(x)
print(x.shape)
print(y.shape)
print(y)
