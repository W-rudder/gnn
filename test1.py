import torch
import numpy as np

# a = torch.randint(6,10,(2,3,4))
# print(a)
# mask = torch.tensor([[1,0,1],[0,0,0]])
# mask = mask.reshape([2, 3, 1])
# print(mask.shape)
# b = a.masked_fill(mask=mask, value=0)
# print(b)
a = np.array([1,2,3])
b = np.array([1,2,3])
print(set(list(a)+list(b)).remove(1))
