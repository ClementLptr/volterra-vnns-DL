import torch
import torch.nn as nn
from config.logger import setup_logger
from typing import Tuple
from torch.nn import functional as F

logger = setup_logger()

class RKHS_VNN(nn.Module):
    """
    Improved Reproducing Kernel Hilbert Space Volterra Neural Network.
    
    Enhancements:
    - Memory-efficient neighborhood extraction
    - Adaptive layer sizing
    - Improved regularization
    - Dynamic fully connected layers (removed)
    - Multiple kernel types support
    """
    
    def __init__(
        self, 
        num_classes: int, 
        num_ch: int = 3, 
        input_shape: Tuple[int, int, int] = (16, 112, 112),
        dropout_rate: float = 0.5,
        pretrained: bool = False
    ) -> None:
        super(RKHS_VNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        logger.debug("Initializing architecture configuration.")
        # Improved architecture configuration with adaptive sizing
        self.architecture_config = {
            'layer1': {'out_channels': 24, 'kernel_size': 3, 'stride': 1},
            # 'layer2': {'out_channels': 32, 'kernel_size': 3, 'stride': 2},
        #     'layer3': {'out_channels': 64, 'kernel_size': 3, 'stride': 1},
        #     'layer4': {'out_channels': 96, 'kernel_size': 3, 'stride': 2},
        }
                
        self.num_channels = num_ch
        self.dropout_rate = dropout_rate
        
        logger.debug("Calculating feature dimensions dynamically.")
        # Calculate feature dimensions dynamically
        self.feature_dims = self._calculate_feature_dims()
        
        logger.debug("Initializing layers.")
        self._initialize_layers()
        self._initialize_weights()
        
        if pretrained:
            logger.debug("Loading pretrained weights.")
            self._load_pretrained_weights()

    def _calculate_feature_dims(self) -> int:
        """Calculate output dimensions after convolutions dynamically."""
        d, h, w = self.input_shape
        
        logger.debug(f"Initial input dimensions: {d}x{h}x{w}")
        for layer_config in self.architecture_config.values():
            stride = layer_config['stride']
            padding = layer_config['kernel_size'] // 2
            
            d = (d + 2 * padding - layer_config['kernel_size']) // stride + 1
            h = (h + 2 * padding - layer_config['kernel_size']) // stride + 1
            w = (w + 2 * padding - layer_config['kernel_size']) // stride + 1
            logger.debug(f"After layer with stride {stride}: {d}x{h}x{w}")
            
        return d * h * w * self.architecture_config['layer1']['out_channels']

    def _initialize_layers(self) -> None:
        """Initialize network layers with improved structure."""
        
        self.eta_params = nn.ParameterDict()
        for i in range(1, len(self.architecture_config) + 1):
            # Get neighborhood size for the layer
            neighborhood_size = self.architecture_config[f'layer{i}']['kernel_size']
            neighborhood = neighborhood_size ** 3  # Calcul du nombre d'éléments dans le voisinage (pour un noyau 3x3x3, c'est 27)
            
            # Initialize eta with the shape [batch_size, channels, neighborhood_size, neighborhood_size, depth, height, width]
            eta_shape = (
                self.num_channels,                 # Number of channels (e.g., 3 for RGB)
                neighborhood, 
                neighborhood,                      # Number of neighbors (neighborhood_size^3, e.g., 27 for 3x3x3)
                self.input_shape[0],               # Depth (T)
                self.input_shape[1],               # Height (H)
                self.input_shape[2],               # Width (W)
            )

            # Initialize eta as a learnable parameter (scaled by the number of channels)
            self.eta_params[f'eta_{i}'] = nn.Parameter(
                torch.ones(eta_shape, dtype=torch.float32) / eta_shape[0]
            )
            logger.debug(f"Initialized eta_{i} with shape: {self.eta_params[f'eta_{i}'].shape}")
        
        # Initialize batch normalization and dropout layers
        self.bn_layers = nn.ModuleDict()
        self.dropout_layers = nn.ModuleDict()
        
        for i in range(1, len(self.architecture_config) + 1):
            out_channels = self.architecture_config[f'layer{i}']['out_channels']
            self.bn_layers[f'bn_{i}'] = nn.BatchNorm3d(out_channels)
            self.dropout_layers[f'dropout_{i}'] = nn.Dropout3d(p=self.dropout_rate)
            logger.debug(f"Initialized BatchNorm3d and Dropout3d for layer {i}")


    def _poly_kernel(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Compute the inhomogeneous polynomial kernel for a single input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, channels, kernel_size^3, depth, height, width]
            degree: Degree of the polynomial kernel

        Returns:
            Kernel matrix of shape [batch_size, kernel_size^3, kernel_size^3, depth, height, width]
        """
        batch_size, channels, neighborhood_size, _, _, _ = x.size()

        # Flatten the neighborhood (kernel_size^3) dimension
        x_flat = x.view(batch_size, channels, neighborhood_size, -1)  # Shape: [batch_size, channels, neighborhood_size, depth * height * width]

        # Compute the dot product of each element in the neighborhood with every other element in the same neighborhood
        dot_product = torch.matmul(x_flat, x_flat.transpose(2, 3))  # Shape: [batch_size, channels, neighborhood_size, neighborhood_size, depth, height, width]

        # Apply the polynomial kernel function (1 + dot_product) ** degree
        return (1 + dot_product).pow(degree)

    def _extract_neighborhood(self, x: torch.Tensor, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> torch.Tensor:
        """
        Extract a neighborhood for each pixel/voxel using sliding windows, optimized for memory.

        Args:
            x: Input tensor [batch_size, channels, depth, height, width]
            kernel_size: Size of the 3D window
            stride: Stride of the window
            padding: Padding to add around the tensor

        Returns:
            Extracted patches tensor [batch_size, channels, kernel_size^3, depth', height', width']
        """
        # Apply padding (correct padding order: left, right, top, bottom, front, back)
        x_padded = nn.functional.pad(x, (padding, padding, padding, padding, padding, padding), mode='constant', value=0)
        logger.debug(f"Padded input tensor shape: {x_padded.shape}")
        
        # Use unfold to extract 3D neighborhoods, but directly work with unfolded indices
        unfolded = x_padded.unfold(2, kernel_size, stride)  # Depth dimension
        unfolded = unfolded.unfold(3, kernel_size, stride)  # Height dimension
        unfolded = unfolded.unfold(4, kernel_size, stride)  # Width dimension
        
        # Now, we want to reshape unfolded into a form where we keep neighborhood information
        patches = unfolded.contiguous().view(
            x.size(0), x.size(1), -1, unfolded.size(2), unfolded.size(3), unfolded.size(4)
        )
        logger.debug(f"Extracted patches shape: {patches.shape}")
        
        # Return the patches for each voxel/pixel
        return patches
    
    
    def _volterra_approximation(self, x: torch.Tensor, y: torch.Tensor, eta: torch.Tensor, degree: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> torch.Tensor:
        """
        Volterra series approximation using a polynomial kernel applied between elements in the neighborhood.

        Args:
            x: Input tensor [batch_size, channels, depth, height, width]
            y: The corresponding target tensor [batch_size, channels, depth, height, width]
            eta: Layer-specific eta parameters for regularization [channels, neighborhood_size, neighborhood_size, depth, height, width]
            degree: Degree of the polynomial kernel
            kernel_size: Size of the 3D window (patch)
            stride: Stride of the sliding window
            padding: Padding to apply around the tensor

        Returns:
            Tensor after Volterra approximation
        """
        batch_size, channels, depth, height, width = x.size()

        # Step 1: Extract the neighborhoods using the provided function
        x_patches = self._extract_neighborhood(x, kernel_size=kernel_size, stride=stride, padding=padding)  # Shape: [batch_size, channels, kernel_size^3, depth', height', width']
        logger.debug(f"Extracted patches for Volterra approximation, shape: {x_patches.shape}")

        # Step 2: Prepare eta for efficient calculation (eta does not depend on batch size)
        # Repeat eta for each sample in the batch
        eta_repeated = eta.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1, 1)  # Shape: [batch_size, channels, neighborhood_size, neighborhood_size, depth, height, width]

        # Step 3: Flatten x_patches for polynomial kernel computation
        x_patches = x_patches.view(batch_size, channels, -1, depth, height, width)  # Shape: [batch_size, channels, kernel_size^3, depth, height, width]
        
        logger.debug(f"Flattened x_patches shape: {x_patches.shape}")

        # Step 4: Compute the polynomial kernels for each neighborhood in the batch
        # Apply polynomial kernel to all elements in the neighborhood without pairwise difference
        kernel_values = self._poly_kernel(x_patches, degree)  # Apply polynomial kernel to the neighborhood elements
        logger.debug(f"Kernel values shape after applying polynomial kernel: {kernel_values.shape}")

        # Step 5: Apply eta for regularization and sum over the neighborhood
        kernel_values = kernel_values * eta_repeated
        output = kernel_values.sum(dim=2)  # Sum over the neighborhood dimension

        logger.debug(f"Output after regularization: {output.shape}")

        return output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RKHS_VNN model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, depth, height, width]
        
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        logger.debug("Starting forward pass.")
        
        # Extract feature maps using convolution layers
        for i in range(1, len(self.architecture_config) + 1):
            x = self._volterra_approximation(
                x,
                x, 
                self.eta_params[f'eta_{i}'],
                degree=2  # Using degree=2 for polynomial kernel approximation
            )
            
            logger.debug(f"Shape after Volterra approximation for layer {i}: {x.shape}")
            
            # Apply batch normalization and dropout
            x = self.bn_layers[f'bn_{i}'](x)
            x = self.dropout_layers[f'dropout_{i}'](x)
            logger.debug(f"Shape after batch normalization and dropout for layer {i}: {x.shape}")
        
        # Final output layer
        x = x.view(x.size(0), -1)  # Flatten the output tensor
        x = self.fc(x)
        return x
    
    def _initialize_weights(self) -> None:
        """Enhanced weight initialization with layer-specific schemes."""
        logger.debug("Initializing weights for all layers.")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_1x_lr_params(model: RKHS_VNN):
    """
    Returns parameters that should use the base learning rate.
    
    This includes:
    - eta parameters (Volterra approximation weights)
    - batch normalization parameters
    - final classification layers
    
    Args:
        model: The RKHS_VNN model instance
    
    Returns:
        Iterator over model parameters that should use 1x learning rate
    """
    # Parameters of batch normalization layers
    for layer in model.bn_layers.values():
        for param in layer.parameters():
            if param.requires_grad:
                yield param
    
    # Eta parameters for Volterra approximation
    for eta in model.eta_params.values():
        if eta.requires_grad:
            yield eta

if __name__ == '__main__':
    # Initialisation du modèle
    logger.debug("Initializing the RKHS_VNN model.")
    model = RKHS_VNN(
        num_classes=10,           # Exemple: 10 classes de sortie
        num_ch=3,                 # Exemple: 3 canaux pour l'entrée (RGB)
        input_shape=(16, 112, 112),  # Exemple de taille d'entrée (profondeur, hauteur, largeur)
        dropout_rate=0.5,         # Taux de dropout
        pretrained=False          # Pas de poids pré-entrainés
    )

    # Création d'un tensor d'entrée aléatoire (par exemple, un lot de 8 échantillons)
    input_tensor = torch.randn(8, 3, 16, 112, 112)  # [batch_size, channels, depth, height, width]
    logger.debug(f"Input tensor shape: {input_tensor.shape}")

    # Passage avant dans le modèle
    output = model(input_tensor)
    logger.debug(f"Model output shape: {output.shape}")

    # Exemple : Afficher la sortie
    print("Model output:", output)
    