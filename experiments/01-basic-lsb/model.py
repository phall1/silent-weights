"""Simple neural network model for steganography proof-of-concept."""

from typing import Dict
import torch
import torch.nn as nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for SimpleClassifier."""
    input_size: int = 784
    hidden_size: int = 128
    num_classes: int = 10
    seed: int = 42


class SimpleClassifier(nn.Module):
    """Simple feedforward classifier for steganography experiments."""
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc3 = nn.Linear(config.hidden_size // 2, config.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_dummy_model(config: ModelConfig = ModelConfig()) -> SimpleClassifier:
    """Create model with realistic weights for embedding experiments."""
    torch.manual_seed(config.seed)
    model = SimpleClassifier(config)
    
    # Minimal training for realistic parameter distributions
    dummy_input = torch.randn(32, config.input_size)
    dummy_target = torch.randint(0, config.num_classes, (32,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(10):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def get_parameter_info(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """Get detailed parameter information for embedding capacity planning."""
    info = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_bytes = param_count * 4  # 32-bit floats
        total_params += param_count
        
        info[name] = {
            "shape": list(param.shape),
            "params": param_count,
            "bytes": param_bytes
        }
    
    info["total"] = {"params": total_params, "bytes": total_params * 4}
    return info


if __name__ == "__main__":
    config = ModelConfig()
    model = create_dummy_model(config)
    
    info = get_parameter_info(model)
    print(f"Model has {info['total']['params']:,} parameters ({info['total']['bytes']:,} bytes)")
    
    for name, data in info.items():
        if name != "total":
            print(f"  {name}: {data['params']:,} params")