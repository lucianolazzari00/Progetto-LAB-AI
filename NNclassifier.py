import torch
import torch.nn as nn
import torch.nn.functional as F 

class ClassifierEasy(nn.Module):
    def __init__(self,n_in,n_hid,n_out):
        super().__init__()
        self.linear1 = nn.Linear(n_in,n_hid)
        self.act_f = nn.Tanh()
        self.linear2 = nn.Linear(n_hid,n_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_f(x)
        x = self.linear2(x)
        return x

model = ClassifierEasy(2,4,1)
print(model)

for name,param in model.named_parameters():
    #prova
    print(name)
    print("--")
    print(param.shape)












### Tutorial operazioni pytorch
# x = torch.randn(1,5, dtype=torch.float32, requires_grad=True)
# z = torch.randn(5,1)
# print(x)
# a=x+2
# b= a**2
# y = torch.matmul(x,z)
# print(y)
# y.backward()
# print(x.grad)
# print(torch.cuda.is_available())

print("--------------------")
x = torch.arange(3,dtype=torch.float32, requires_grad=True)
a = x+2
b = a**2
c = b + 3
y = c.mean()
print(y)
y.backward()
print(x.grad)