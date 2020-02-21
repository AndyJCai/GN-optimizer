import torch
import torchvision
import numpy as np
from typing import List

'''
Gaussian-Newton method is simply approximating Newton's method by replacing the Hessian with 
is approximated by J^T.J
'''

'''
Gauss-Newton algorithm in a few steps:

1. Create an array of unknown parameters in the function:
b = (b1, b2,...,bn)

Initialise the parameters and for each data point in matrix x, calculate the predicted values (y').
Calculate the residuals:
ri = y'i - yi

Find the partial differential of residuals with respect to the parameters and generate Jacobian matrix.
Following an iterative process, calculate the new values for the parameters using the following equation:
s is the iteration number, J is the Jacobian matrix, JT is the transpose of J.

reference: https://www.codeproject.com/Articles/1175992/Implementation-of-Gauss-Newton-Algorithm-in-Java
'''

class Gauss_Newton:

    def resid(x: torch.tensor, y: torch.tensor, b: torch.tensor) -> torch.tensor:
        res = torch.tensor((len(y), 1), dtype=torch.float32)

        for i in range(len(y)):
            res[i][0] = test_func(x[i][0], b) - y[i]

        return res



    def test_func(self, x: torch.tensor, b: torch.tensor):
        '''
        @param: x - The input x array
        @param: b - The coefficients in the function
        @return: y - The result
        '''

        # y = (x * a1) / (a2 + x)
        self.x, self.b = x, b
        return (x * b[0]) / (b[1] + x)

    def get_jacobian(net, x, noutputs):
        x = x.squeeze()
        n = x.size()[0]
        x = x.repeat(noutputs, 1)
        x.requires_grad_(True)
        y = net(x)
        y.backward(torch.eye(noutputs))
        return x.grad.data


    def trans_jacob(self, jacob: torch.tensor, res: torch.tensor):
        '''
        Perform (J JT)-1 JT r operations:
        '''
        J = jacob
        JT = jacob.t()
        inverse = torch.mm(J, JT).inverse()

        return torch.mm(inverse, JT, res)

    def optimize():
        # TODO: finish optimize function
        pass



if __name__ == "__main__":
    pass