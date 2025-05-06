import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== LoRA Layer ========== #
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Base frozen layer (will be skipped in training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # frozen

        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base_out = F.linear(x, self.weight)
        lora_out = self.dropout(x) @ self.lora_A.T
        lora_out = lora_out @ self.lora_B.T
        return base_out + self.scaling * lora_out

# ========== Apply LoRA ========== #
def apply_lora(model, r=16, alpha=32, dropout=0.05):
    """
    Replaces Q and V projection layers in the model with LoRA-injected layers.
    Freezes all other parameters except lora_A and lora_B.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("wq" in name or "wv" in name):
            in_features = module.in_features
            out_features = module.out_features
            lora_layer = LoRALinear(in_features, out_features, r, alpha, dropout)
            lora_layer.weight.data = module.weight.data.clone()
            
            parent = get_parent(model, name)
            setattr(parent, name.split('.')[-1], lora_layer)

    # Freeze all model params except LoRA adapters
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    return model

# Helper: get parent module for attribute replacement
def get_parent(model, module_name):
    parts = module_name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    return model
