import torch
import rich

# Set PyTorch configuration
torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

# Check the default data type
rich.print("Default data type:", torch.get_default_dtype())
