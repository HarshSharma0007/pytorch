import torch
# create a tensor with specific values
# This code initializes a 2D tensor with specific values.

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)






print(my_tensor.dtype)  # Output: torch.float32
print(my_tensor.device)  # Output: cuda:0 or cpu depending on availability
print(my_tensor.shape)  # Output: torch.Size([2, 3])
print(my_tensor.requires_grad)  # Output: True
print(my_tensor)  # Output: tensor([[1., 2., 3.],

x = torch.empty(size=(3,3))
print(x)  # Output: tensor with uninitialized values
x = torch.zeros((3,3))
print(x)  # Output: tensor filled with zeros
x = torch.rand((3,3))
print(x)  # Output: tensor filled with random values
x = torch.ones((3,3))
print(x)  # Output: tensor filled with ones
x = torch.eye(5,5) # since eye sounds like `I`
print(x)  # Output: identity matrix of size 3x3
x = torch.arange(start=0, end=10, step=2)
print(x)  # Output: tensor([0, 2, 4, 6, 8])
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)  # Output: tensor with 10 values linearly spaced between 0.1 and 1

x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
print(x)  # Output: tensor with values drawn from a normal distribution
x = torch.empty(size=(1,5)).uniform_(0, 1)
print(x)  # Output: tensor with values drawn from a uniform distribution

x = torch.diag(torch.ones(3))
print(x)  # Output: diagonal matrix with ones on the diagonal

# HOw to initialize and convert a tensor to a specific data type

tensor = torch.arange(4)
print(tensor)  # Output: tensor([0, 1, 2, 3])


print(tensor.bool())
print(tensor.short())
print(tensor.int())
print(tensor.long())
print(tensor.half()) 
print(tensor.float())
print(tensor.double())

# Array to tensor conversion and vice versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
print(tensor)  # Output: tensor filled with zeros