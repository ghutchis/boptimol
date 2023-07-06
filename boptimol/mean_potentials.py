from gpytorch.means import ConstantMean, Mean
import torch

class MorseMean(Mean):
    
    """ A configurable mean function using the Morse potential

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        batch_shape: train data
        bias: an energy offset term
    Returns:
        Energy using Morse Potential
    """
    
    def __init__(self, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="depth", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        self.register_parameter(name="center", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        self.register_parameter(name="alpha", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        position = x - self.center
        res = torch.exp(-1* self.alpha * position)
        res = self.depth * (1 - res)**2
        if self.bias is not None:
            res = res + self.bias
        res.view(-1)
        return res

    
class QuadraticMean(Mean):
    
    """ A configurable mean function modelled as a Quadratic

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        batch_shape: train data
        bias: an energy offset term
    Returns:
        Energy by a Mean Function
    """
    
    def __init__(self, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        self.register_parameter(name="center", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        offset_sq = (x - self.center)**2
        res = torch.mul(self.weights, offset_sq)
        if self.bias is not None:
            res = res + self.bias
        return res.sum(axis=1).squeeze()