
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # Basic residual block for ResNet architecture
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        # Initialize basic residual block
        super(BasicBlock, self).__init__()
        
        # First convolution layer: 3x3 conv with potential stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer: 3x3 conv with stride 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsampling layer for dimension matching
        self.downsample = downsample
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Forward propagation
        # Save residual connection input
        identity = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = F.relu(out)
        
        return out


class ResNet18(nn.Module):

    def __init__(self, num_classes=10):
        
        super(ResNet18, self).__init__()
        
        # Network layer channel numbers
        self.in_channels = 64
        
        # Initial convolution layer: convert 3-channel RGB images to 64 channels
        # Fashion-MNIST resized to 224x224 with 3 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four stages of ResNet
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   # 2 residual blocks
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 2 residual blocks, stride=2
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 2 residual blocks, stride=2
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 2 residual blocks, stride=2
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Dropout layer for final classification regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        
        # Build one stage of ResNet (containing multiple residual blocks)
        downsample = None
        
        # Need downsampling if stride != 1 or input/output channels don't match
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # First residual block, may include downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining residual blocks, all with stride=1
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        # Forward propagation
        # Input shape: (batch_size, 3, 224, 224)
        
        # Initial convolution and pooling
        x = self.conv1(x)        # (batch_size, 64, 112, 112)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)      # (batch_size, 64, 56, 56)
        
        # Four stages of ResNet
        x = self.layer1(x)       # (batch_size, 64, 56, 56)
        x = self.layer2(x)       # (batch_size, 128, 28, 28)
        x = self.layer3(x)       # (batch_size, 256, 14, 14)
        x = self.layer4(x)       # (batch_size, 512, 7, 7)
        
        # Global average pooling
        x = self.avgpool(x)      # (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 512)
        
        # Add dropout and perform classification
        x = self.dropout(x)
        x = self.fc(x)           # (batch_size, num_classes)
        
        return x


def create_resnet18(num_classes=10):
    
    # create new ResNet-18 model from ground up
    model = ResNet18(num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Test model
    print("Testing ResNet-18 model...")
    
    # Create model instance
    model = create_resnet18(num_classes=10)
    
    # Create test input (simulating Fashion-MNIST batch with 224x224 size)
    test_input = torch.randn(4, 3, 224, 224)  # batch_size=4
    
    # Forward propagation test
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("ResNet-18 model test successful!")
