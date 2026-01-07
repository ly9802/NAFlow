# -- coding:utf-8 --
# Time:2025/12/25 19:11
import torch
import torch.nn as nn
from torch.nn import functional as F

def apply_inverse_lsmr(A, Y, atol=1e-8, btol=1e-8):
    from scipy.sparse.linalg import lsmr
    x, *_=lsmr(A,Y,atol=atol,btol=btol)
    return x

def one_inverse(jacobian_matrix, y):
    A=jacobian_matrix.squeeze(dim=0).detach().cpu().numpy()
    Y=y.squeeze(dim=0).detach().cpu().numpy();
    x=apply_inverse_lsmr(A,Y)
    return torch.from_numpy(x).to(y.device);

def batch_jacobian(x, y):
    batch_size, n = x.shape
    m = y.size(dim=-1)
    jac = torch.zeros(batch_size, m, n, device=x.device)
    for i in range(m):
        grad_outputs = torch.zeros_like(y,device=x.device)
        grad_outputs[:, i] = 1.0
        grad_i = torch.autograd.grad(outputs=y,inputs=x,allow_unused=True,grad_outputs=grad_outputs,retain_graph=True,create_graph=False)[0]
        if grad_i==None:
            jac[:, i, :] = torch.zeros(size=(batch_size,n),device=x.device)
        else:
            jac[:,i,:]=grad_i
    if m>n:
        Jac=jac[:, :n, :]
        return Jac
    return jac

class ReverseReLU(nn.Module):
    def __init__(self,relu, inp,out):
        super(ReverseReLU,self).__init__()
        self.module=relu
        self.input=inp;
        self.output=out
    def forward(self,x,grad=None):
        y=x;
        if grad==None:
            return y, None
        else:
            grad_tuple=torch.autograd.grad(outputs=self.output, inputs=self.input, retain_graph=True,grad_outputs=grad,allow_unused=True,materialize_grads=True)
            selected_index = torch.ne(grad_tuple[0], 0).float().to(self.input.device)
            return torch.mul(y, selected_index), grad_tuple[0]
class ReverseBatchNorm(nn.Module):
    def __init__(self,batchnorm,inp,out):
        super(ReverseBatchNorm,self).__init__()
        self.module=batchnorm
        self.bias=batchnorm.bias.data.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        #(num_channel,)-->(1,num_channel,1,1)
        self.weight=batchnorm.weight.data.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.mean=batchnorm.running_mean.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) #(num_channel,)
        self.var=batchnorm.running_var.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) #(num_channel,)
        self.momentum=batchnorm.momentum
        self.eps=batchnorm.eps
        self.input=inp;
        self.output=out;
    def forward(self,x,grad=None):
        x=torch.sub(x,self.bias)
        x=torch.div(x,self.weight)
        var=torch.add(self.var,self.eps)
        x=torch.mul(x,torch.sqrt(var))
        y=torch.add(x,self.mean)
        if grad==None:
            return y, None
        else:
            grad_tuple = torch.autograd.grad(outputs=self.output, inputs=self.input, retain_graph=True, grad_outputs=grad,allow_unused=True,materialize_grads=True)
            selected_index = torch.ne(grad_tuple[0], 0).float().to(self.input.device)
            return torch.mul(y, selected_index),grad_tuple[0]

class ReverseMaxPool2d(nn.Module):
    def __init__(self,max_pool, inp,out):
        super(ReverseMaxPool2d,self).__init__()
        self.module=max_pool
        self.kernel_size=max_pool.kernel_size
        self.stride=max_pool.stride
        self.padding=max_pool.padding
        self.dilation=max_pool.dilation
        self.ceil_mode=max_pool.ceil_mode
        self.input=inp
        self.output=out
        self.uppool=nn.MaxUnpool2d(kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

    def forward(self,x,grad=None):
        out, index = F.max_pool2d(self.input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode,return_indices=True)
        y = self.uppool(x, index, self.input.size())
        if grad==None:
            return y, None
        else:
            grad_tuple=torch.autograd.grad(outputs=self.output, inputs=self.input, retain_graph=True,grad_outputs=grad,allow_unused=True,materialize_grads=True)
            selected_index=torch.ne(grad_tuple[0], 0).float().to(self.input.device)
            return torch.mul(y, selected_index), grad_tuple[0]

class ReverseComplex(nn.Module):
    def __init__(self,complex_module, inp, out):
        super(ReverseComplex,self).__init__()
        self.module=complex_module
        self.input=inp
        self.output=out
        self.num_y=self.output.size(-1)*self.output.size(-2)*self.output.size(-3);
        self.num_x=self.input.size(-1)*self.input.size(-2)*self.input.size(-3);
    def forward(self,x, grad=None):
        batch_size, dim, h,w=self.input.shape
        bias = self.module(torch.zeros_like(self.input).to(self.input.device))
        x_ = torch.sub(x, bias).contiguous().view(batch_size, -1)
        num_y=x_.size(dim=-1)
        if num_y>self.num_x:
            x_ = x_[:, :self.num_x]
        jacobian_matrix = batch_jacobian(self.input.contiguous().view(batch_size,-1), self.output.contiguous().view(batch_size, -1)) #(bs, p,p)
        y=one_inverse(jacobian_matrix,x_)
        y=y.contiguous().view(batch_size,dim,h,-1)
        if y.abs().max()==0:
            y=y+self.input
        if grad==None:
            return self.input, None
        else:
            grad_tuple = torch.autograd.grad(outputs=self.output, inputs=self.input, retain_graph=True, grad_outputs=grad, allow_unused=True, materialize_grads=True)
            selected_index = torch.ne(grad_tuple[0], 0).float().to(self.input.device)
        return torch.mul(y, selected_index),grad_tuple[0];

class ReverseConv(nn.Module):
    def __init__(self,convolution_layer,inp, out):
        super(ReverseConv,self).__init__()
        self.module=convolution_layer
        self.in_channels=convolution_layer.in_channels
        self.out_channels = convolution_layer.out_channels
        self.kernel_size = convolution_layer.kernel_size
        self.stride = convolution_layer.stride
        self.padding = convolution_layer.padding
        self.dilation = convolution_layer.dilation
        self.weight = convolution_layer.weight.data
        if convolution_layer.bias == None:
            self.bias = torch.zeros((1, self.out_channels, 1, 1), device=convolution_layer.weight.device)
        else:
            self.bias = convolution_layer.bias.data.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.groups = convolution_layer.groups
        self.input = inp
        self.output = out
        self.num_y = self.output.size(-1) * self.output.size(-2) * self.output.size(-3);
        self.num_x = self.input.size(-1) * self.input.size(-2) * self.input.size(-3);
    def forward(self,x,grad=None):
        batch_size, dim,h,w=self.input.shape
        x_ = torch.sub(x, self.bias).contiguous().view(batch_size, -1)
        num_y = x_.size(dim=-1)
        if num_y > self.num_x:
            x_ = x_[:, :self.num_x]
        jacobian_matrix=batch_jacobian(self.input.contiguous().view(batch_size,-1), self.output.contiguous().view(batch_size,-1))
        y = one_inverse(jacobian_matrix, x_)
        y = y.contiguous().view(batch_size,dim, h, -1)
        if y.abs().max()==0:
            y=y+self.input
        if grad==None:
            return y, None
        else:
            grad_tuple = torch.autograd.grad(outputs=self.output, inputs=self.input, retain_graph=True,grad_outputs=grad,allow_unused=True,materialize_grads=True)
            selected_index = torch.ne(grad_tuple[0], 0).float().to(y.device)
            return torch.mul(y, selected_index),grad_tuple[0]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("well done!")