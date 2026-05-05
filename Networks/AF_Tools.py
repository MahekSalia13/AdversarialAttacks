
import torch.nn as nn


Activation_Functions_ = [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.SELU(), nn.CELU(), nn.Mish(), nn.GELU()]
operators_ = [0.25, 0.5, 0.75, 1, 2, 3, 4]   #  < 2 : a.f(x) , 2 : f(g(x)) , 3 : f(x)+g(x) , 4 : f(x)*g(x)
operators_p_ = [0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.25]   # probability
