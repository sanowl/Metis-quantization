import torch
import torch.nn as nn


class DualRangeRegularization:
    """
    Implements dual-range regularization from Metis paper.
    
    Formula: R(W) = λ₁∑Wᵢ² + λ₂∑1/(Wᵢ² + ε)
    
    Where:
    - λ₁ controls penalty on large magnitudes (prevents overflow)
    - λ₂ penalizes near-zero values (prevents underflow) 
    - ε is smoothing term for numerical stability
    """
    
    def __init__(self, lambda1=1e-4, lambda2=1e-6, epsilon=1e-8):
        """
        Args:
            lambda1: Penalty weight for large magnitudes
            lambda2: Penalty weight for near-zero values  
            epsilon: Smoothing term for numerical stability
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon
    
    def __call__(self, parameters):
        """
        Compute dual-range regularization loss for given parameters.
        
        Args:
            parameters: Model parameters (can be single tensor or iterable of tensors)
            
        Returns:
            regularization_loss: Scalar tensor
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        total_loss = 0.0
        
        for param in parameters:
            if param.requires_grad:
                # L2 penalty for large magnitudes: λ₁∑Wᵢ²
                l2_term = self.lambda1 * torch.sum(param ** 2)
                
                # Inverse penalty for near-zero values: λ₂∑1/(Wᵢ² + ε)
                inverse_term = self.lambda2 * torch.sum(1.0 / (param ** 2 + self.epsilon))
                
                total_loss += l2_term + inverse_term
        
        return total_loss
    
    def set_lambdas(self, lambda1=None, lambda2=None):
        """Update regularization weights."""
        if lambda1 is not None:
            self.lambda1 = lambda1
        if lambda2 is not None:
            self.lambda2 = lambda2


class MetisLoss(nn.Module):
    """
    Combined loss function that includes task loss + dual-range regularization.
    """
    
    def __init__(self, task_loss_fn, dual_range_reg=None):
        """
        Args:
            task_loss_fn: Primary task loss function (e.g., CrossEntropyLoss)
            dual_range_reg: DualRangeRegularization instance (optional)
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.dual_range_reg = dual_range_reg or DualRangeRegularization()
    
    def forward(self, predictions, targets, model_parameters=None):
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model_parameters: Model parameters for regularization (optional)
            
        Returns:
            total_loss: Task loss + regularization loss
            task_loss: Task loss only (for logging)
            reg_loss: Regularization loss only (for logging)
        """
        # Compute primary task loss
        task_loss = self.task_loss_fn(predictions, targets)
        
        # Compute regularization loss if parameters provided
        if model_parameters is not None:
            reg_loss = self.dual_range_reg(model_parameters)
            total_loss = task_loss + reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=task_loss.device)
            total_loss = task_loss
        
        return total_loss, task_loss, reg_loss


def get_linear_layer_parameters(model):
    """
    Extract parameters from linear layers (BitLinear, LinearLowbit, nn.Linear) for regularization.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of parameter tensors from linear layers
    """
    from .bitlinear import BitLinear, LinearLowbit
    
    linear_params = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, BitLinear, LinearLowbit)):
            if hasattr(module, 'weight') and module.weight is not None:
                linear_params.append(module.weight)
            # Also include SVD components if they exist
            if hasattr(module, 'ulinear') and module.ulinear is not None:
                if hasattr(module.ulinear, 'weight'):
                    linear_params.append(module.ulinear.weight)
            if hasattr(module, 'vlinear') and module.vlinear is not None:
                if hasattr(module.vlinear, 'weight'):
                    linear_params.append(module.vlinear.weight)
    
    return linear_params