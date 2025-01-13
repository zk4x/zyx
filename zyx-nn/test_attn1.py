import math
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter

class CausalSelfAttention:
    def __init__(self, n_embd: int, n_head: int, bias: bool, dtype):
        self.c_attn = Linear(n_embd, 3*n_embd, bias=bias)
        self.c_proj = Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head

    def forward(self, x: Tensor):
        b, t, c = x.shape
        q, k, v = self.c_attn.forward(x).split([c, c, c], 2)

        k = k.reshape([b, t, self.n_head, c // self.n_head]).transpose(1, 2)
        q = q.reshape([b, t, self.n_head, c // self.n_head]).transpose(1, 2)
        v = v.reshape([b, t, self.n_head, c // self.n_head]).transpose(1, 2)

        scale = 1.0 / math.sqrt(k.size(-1))
        att = q.matmul(k.transpose(-2, -1)) * scale

        #print(f"{scale=}, att: {att}")

        att = att.softmax(-1)
        y = att.matmul(v)
        y = y.transpose(1, 2).reshape([b, t, c])
        y = self.c_proj.forward(y)

        return y

torch.manual_seed(34932049)
torch.set_printoptions(threshold=10000, precision=8)

n_embd = 4
n_head = 2
bias = False
dtype = torch.float32
attn = CausalSelfAttention(n_embd, n_head, bias, dtype)

attn.c_attn.weight = Parameter(torch.tensor([[-0.495788, 0.119697, -0.139357, 0.059328],
                [0.407094, -0.065494, -0.129729, -0.074552],
                [0.324870, 0.155732, 0.297099, -0.412060],
                [0.020193, -0.336263, -0.009602, 0.116321],
                [-0.453359, -0.220178, 0.232500, 0.120824],
                [-0.457052, -0.312347, -0.267674, 0.344709],
                [-0.262033, -0.192330, -0.090726, -0.405672],
                [-0.472127, -0.110653, -0.040921, -0.487143],
                [-0.459970, 0.357617, 0.109131, 0.214290],
                [0.296274, 0.091488, 0.121792, -0.081484],
                [-0.097352, -0.116311, -0.033035, 0.236983],
                [0.078229, 0.294886, 0.363787, -0.383411]]))
attn.c_attn.bias = None
attn.c_proj.weight = Parameter(torch.tensor([[-0.202461, -0.263050, -0.244990, 0.044416],
                [-0.398643, 0.219820, 0.253934, 0.204294],
                [-0.323065, 0.195841, -0.106940, 0.142828],
                [0.233007, -0.026790, -0.293228, 0.118043]]))
attn.c_proj.bias = None

#x = torch.randn([1, 3, 4], dtype=torch.float32)

x = torch.tensor([[
    [-1.363837, -0.801618, -1.304842, -1.664811],
    [-0.385430, -0.955608, -1.003842, 0.073811],
    [-0.785831, 1.030346, 0.593785, -0.214361],
]])

#print(attn.c_attn.weight)
#print(attn.c_proj.weight)
#print(x)

for _ in range(5):
    x = attn.forward(x)

print(x.shape)
print(x.dtype)
print(x)
