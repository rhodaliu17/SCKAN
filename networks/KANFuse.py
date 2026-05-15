import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.kan import KANLinear

#v4
class KANFuse(nn.Module):
    """
    KAN-based Prototype Fusion Network
    
    Features:
    - Aggregates batch prototypes using KAN
    - Normalizes prototypes to [0, 1] before processing
    - Uses stable hyperparameters from reference implementation
    """
    
    def __init__(
        self, 
        proto_dim=128,
        num_classes=2,
        n_regions=3,
        hidden_dim=64,
        grid_size=3,
        spline_order=3,
        dropout=0.1,
    ):
        """
        Args:
            proto_dim: Dimension of prototype features
            num_classes: Number of classes
            n_regions: Number of regions per class
            hidden_dim: Hidden dimension for KAN layers
            grid_size: Grid size for B-spline basis
            spline_order: Order of B-spline (usually 3)
            dropout: Dropout rate
        """
        super(KANFuse, self).__init__()
        
        self.proto_dim = proto_dim
        self.num_classes = num_classes
        self.n_regions = n_regions
        self.hidden_dim = hidden_dim
        
        # ===== KAN hyperparameters (from reference NormalConvsKAN) =====
        kan_kwargs = {
            'grid_size': grid_size,
            'spline_order': spline_order,
            'scale_noise': 0.01,        # Smaller initialization noise for stability
            'scale_base': 1.0,
            'scale_spline': 1.0,
            'base_activation': nn.SiLU,
            'grid_eps': 0.02,
            'grid_range': [0, 1],       # Match normalized data range
        }
        
        # ===== NEW: Stage 0 - Batch aggregation =====
        # Aggregate [B, n_regions, dim] → [1, n_regions, dim]
        self.kan_labeled = KANLinear(proto_dim, proto_dim, **kan_kwargs)
        self.kan_unlabeled = KANLinear(proto_dim, proto_dim, **kan_kwargs)
        
        # ===== Stage 1 & 2: Fusion layers (unchanged) =====
        self.kan_fuse1 = KANLinear(2 * proto_dim, hidden_dim, **kan_kwargs)
        self.kan_fuse2 = KANLinear(hidden_dim, proto_dim, **kan_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        
        # For tracking normalization statistics (optional, for debugging)
        self.register_buffer('_debug_mode', torch.tensor(True))
        
    def _normalize_prototype(self, proto):
        """
        Normalize prototype to [0, 1] range using min-max normalization
        
        Args:
            proto: Tensor of shape (n_regions, dim) or (1, dim)
        
        Returns:
            Normalized prototype in [0, 1]
        """
        proto_min = proto.min()
        proto_max = proto.max()
        
        # Avoid division by zero
        if proto_max - proto_min < 1e-8:
            # If all values are the same, map to 0.5
            return torch.ones_like(proto) * 0.5
        
        # Min-max normalization to [0, 1]
        normalized = (proto - proto_min) / (proto_max - proto_min)
        
        return normalized
    
    def _process_prototype_shape(self, proto):
        """
        Process prototype to ensure it's 2D: (n_regions, dim)
        
        Args:
            proto: Input prototype (can be 1D, 2D, or higher)
        
        Returns:
            2D tensor of shape (n_regions, dim)
        """
        # Remove any trailing dimensions of size 1
        while proto.dim() > 2 and proto.shape[-1] == 1:
            proto = proto.squeeze(-1)
        
        # If still higher than 2D, flatten extra dimensions
        if proto.dim() > 2:
            proto = proto.reshape(proto.shape[0], -1)
        
        # If 1D, add batch dimension
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)
        
        return proto
    
    def _aggregate_batch_prototypes(self, proto_batch, kan_layer, update_grid=False):
        """
        Aggregate batch prototypes using KAN
        
        Args:
            proto_batch: Tensor of shape (B, n_regions, dim)
            kan_layer: KANLinear layer to use for aggregation
            update_grid: bool, whether to update KAN grid
        
        Returns:
            Aggregated prototype of shape (n_regions, dim)
        """
        B, n_regions, dim = proto_batch.shape
        
        # Reshape: (B, n_regions, dim) → (B*n_regions, dim)
        proto_flat = proto_batch.reshape(B * n_regions, dim)
        
        # Normalize to [0, 1]
        proto_flat = self._normalize_prototype(proto_flat)
        
        # Pass through KAN
        if update_grid:
            # Sample a subset for update_grid to avoid OOM
            sample_size = min(100, proto_flat.shape[0])
            kan_layer.update_grid(proto_flat[:sample_size])
        
        proto_transformed = kan_layer(proto_flat)  # (B*n_regions, dim)
        
        # Reshape back: (B*n_regions, dim) → (B, n_regions, dim)
        proto_transformed = proto_transformed.reshape(B, n_regions, dim)
        
        # Aggregate over batch dimension (mean pooling)
        proto_aggregated = proto_transformed.mean(dim=0)  # (n_regions, dim)
        
        return proto_aggregated
    
    def forward(self, labeled_prototypes, unlabeled_prototypes, update_grid=False):
        """
        Forward pass: fuse labeled and unlabeled prototypes using KAN
        
        Args:
            labeled_prototypes: List[Tensor], len=num_classes
                Each element shape: (B, n_regions, proto_dim)
            unlabeled_prototypes: List[Tensor], len=num_classes
                Each element shape: (B, n_regions, proto_dim)
            update_grid: bool, whether to update KAN grid (use sparingly)
        
        Returns:
            fused_prototypes: List[Tensor], len=num_classes
                Each element shape: (n_regions, proto_dim)
        """
        fused_prototypes = []
        
        # Debug: print statistics on first iteration
        if self._debug_mode and update_grid:
            self._print_debug_info(labeled_prototypes, unlabeled_prototypes)
            self._debug_mode = torch.tensor(False)
        
        for c in range(self.num_classes):
            # Get prototypes for current class
            lab_proto_batch = labeled_prototypes[c].clone()      # (B, n_regions, dim)
            unlab_proto_batch = unlabeled_prototypes[c].clone()  # (B, n_regions, dim)
            
            # ===== NEW: Stage 0 - Aggregate batch using KAN =====
            # (B, n_regions, dim) → (n_regions, dim)
            lab_proto = self._aggregate_batch_prototypes(
                lab_proto_batch, 
                self.kan_labeled, 
                update_grid=update_grid
            )
            
            unlab_proto = self._aggregate_batch_prototypes(
                unlab_proto_batch, 
                self.kan_unlabeled, 
                update_grid=update_grid
            )
            
            # Now: lab_proto.shape = (n_regions, dim)
            #      unlab_proto.shape = (n_regions, dim)
            
            # ===== Step 2: Normalize to [0, 1] =====
            lab_proto = self._normalize_prototype(lab_proto)
            unlab_proto = self._normalize_prototype(unlab_proto)
            
            # ===== Step 3: Validate dimensions =====
            assert lab_proto.dim() == 2, f"Expected 2D, got {lab_proto.shape}"
            assert unlab_proto.dim() == 2, f"Expected 2D, got {unlab_proto.shape}"
            assert lab_proto.shape[1] == self.proto_dim, \
                f"Dimension mismatch: {lab_proto.shape[1]} vs {self.proto_dim}"
            assert unlab_proto.shape[1] == self.proto_dim, \
                f"Dimension mismatch: {unlab_proto.shape[1]} vs {self.proto_dim}"
            
            # ===== Step 4: Concatenate and fuse =====
            concat_proto = torch.cat([lab_proto, unlab_proto], dim=-1)  # (n_regions, 2*proto_dim)
            
            # Stage 1: Fusion
            if update_grid:
                self.kan_fuse1.update_grid(concat_proto)
            
            fused = self.kan_fuse1(concat_proto)  # (n_regions, hidden_dim)
            fused = self.dropout(fused)
            
            # Stage 2: Project back
            if update_grid:
                self.kan_fuse2.update_grid(fused)
            
            fused = self.kan_fuse2(fused)  # (n_regions, proto_dim)
            
            fused_prototypes.append(fused)
        
        return fused_prototypes
    
    def _print_debug_info(self, labeled_prototypes, unlabeled_prototypes):
        """Print debug information about prototype statistics"""
        print("=" * 60)
        print("KANFuse Debug Info (First Iteration)")
        print("=" * 60)
        
        for c in range(self.num_classes):
            lab_proto = labeled_prototypes[c]
            unlab_proto = unlabeled_prototypes[c]
            
            print(f"\nClass {c}:")
            print(f"  Labeled prototype:")
            print(f"    Shape: {lab_proto.shape}")
            print(f"    Range: [{lab_proto.min():.4f}, {lab_proto.max():.4f}]")
            print(f"    Mean: {lab_proto.mean():.4f}, Std: {lab_proto.std():.4f}")
            
            print(f"  Unlabeled prototype:")
            print(f"    Shape: {unlab_proto.shape}")
            print(f"    Range: [{unlab_proto.min():.4f}, {unlab_proto.max():.4f}]")
            print(f"    Mean: {unlab_proto.mean():.4f}, Std: {unlab_proto.std():.4f}")
        
        print("=" * 60)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute KAN regularization loss
        
        Args:
            regularize_activation: Weight for activation regularization
            regularize_entropy: Weight for entropy regularization
        
        Returns:
            Total regularization loss
        """
        loss = 0.0
        loss += self.kan_labeled.regularization_loss(regularize_activation, regularize_entropy)
        loss += self.kan_unlabeled.regularization_loss(regularize_activation, regularize_entropy)
        loss += self.kan_fuse1.regularization_loss(regularize_activation, regularize_entropy)
        loss += self.kan_fuse2.regularization_loss(regularize_activation, regularize_entropy)
        return loss
    
    def get_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'proto_dim': self.proto_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'n_regions': self.n_regions,
        }


# ===== Example Usage =====
if __name__ == "__main__":
    # Test the KANFuse module
    batch_size = 4
    proto_dim = 128
    num_classes = 2
    n_regions = 3
    
    # Initialize model
    model = KANFuse(
        proto_dim=proto_dim,
        num_classes=num_classes,
        n_regions=n_regions,
        hidden_dim=64,
        grid_size=3,
        dropout=0.1,
    )
    
    print("Model Info:", model.get_info())
    
    # ===== NEW: Create batch prototypes =====
    labeled_prototypes = [
        torch.randn(batch_size, n_regions, proto_dim) * 2 + 1,  # (4, 3, 128)
        torch.randn(batch_size, n_regions, proto_dim) * 3 + 2,
    ]
    
    unlabeled_prototypes = [
        torch.randn(batch_size, n_regions, proto_dim) * 2 + 1,
        torch.randn(batch_size, n_regions, proto_dim) * 3 + 2,
    ]
    
    print("\nInput shapes:")
    for i in range(num_classes):
        print(f"  Class {i} - Labeled: {labeled_prototypes[i].shape}, Unlabeled: {unlabeled_prototypes[i].shape}")
    
    # Forward pass
    fused = model(labeled_prototypes, unlabeled_prototypes, update_grid=False)
    
    print("\nOutput shapes (after aggregation and fusion):")
    for i, proto in enumerate(fused):
        print(f"  Class {i}: {proto.shape}")
    
    # Regularization loss
    reg_loss = model.regularization_loss(regularize_activation=1e-5, regularize_entropy=1e-5)
    print(f"\nRegularization loss: {reg_loss.item():.6f}")