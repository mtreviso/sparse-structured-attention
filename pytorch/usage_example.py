import torch
import torchsparseattn
a = torch.tensor([1, 2.1, 1.9], dtype=torch.double, requires_grad=True)
lengths = torch.tensor([3]).long()
x = torchsparseattn.Fusedmax(alpha=0.1)(a, lengths)
print(x)
x.sum().backward()  # calculate gradients
print(a.grad)
