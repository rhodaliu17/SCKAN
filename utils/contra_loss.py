import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, diversity_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        
    def forward(self, labeled_prototypes, unlabeled_prototypes):
       
        if len(labeled_prototypes) == 0 or len(unlabeled_prototypes) == 0:
            device = labeled_prototypes[0].device if labeled_prototypes else unlabeled_prototypes[0].device
            return torch.tensor(0.0, device=device)
        
        contra_loss = self._center_contrastive_loss(labeled_prototypes, unlabeled_prototypes)
        
        diversity_loss = self._diversity_regularization(labeled_prototypes, unlabeled_prototypes)
        
        total_loss = (1 - self.diversity_weight) * contra_loss + self.diversity_weight * diversity_loss
        
        return total_loss
    
    def _center_contrastive_loss(self, labeled_prototypes, unlabeled_prototypes):
    
        n_classes = len(labeled_prototypes)
        B, K, D = labeled_prototypes[0].shape
        
        labeled_centers = torch.stack([protos.mean(dim=1) for protos in labeled_prototypes], dim=0)  # (n_classes, B, D)
        unlabeled_centers = torch.stack([protos.mean(dim=1) for protos in unlabeled_prototypes], dim=0)
        
        labeled_with_centers = []
        unlabeled_with_centers = []
        
        for c in range(n_classes):
            # (B, K, D) + (B, 1, D) → (B, K+1, D)
            lab_with_center = torch.cat([
                labeled_prototypes[c], 
                labeled_centers[c].unsqueeze(1)
            ], dim=1)
            unlab_with_center = torch.cat([
                unlabeled_prototypes[c], 
                unlabeled_centers[c].unsqueeze(1)
            ], dim=1)
            
            labeled_with_centers.append(lab_with_center)
            unlabeled_with_centers.append(unlab_with_center)
        
        # Shape: (n_classes, B, K+1, D) → (n_classes * B * (K+1), D)
        labeled_flat = torch.cat([p.reshape(-1, D) for p in labeled_with_centers], dim=0)
        unlabeled_flat = torch.cat([p.reshape(-1, D) for p in unlabeled_with_centers], dim=0)
        
        labeled_flat = F.normalize(labeled_flat, p=2, dim=1, eps=1e-8)
        unlabeled_flat = F.normalize(unlabeled_flat, p=2, dim=1, eps=1e-8)
        
        all_protos = torch.cat([labeled_flat, unlabeled_flat], dim=0)  # (2 * n_classes * B * (K+1), D)
        
        sim_matrix = torch.matmul(all_protos, all_protos.T) / self.temperature
        
        K_plus_1 = K + 1
        total_per_source = n_classes * B * K_plus_1
        
        class_labels = torch.repeat_interleave(
            torch.arange(n_classes), 
            B * K_plus_1
        )
        class_labels = torch.cat([class_labels, class_labels]).to(sim_matrix.device)
        
        pos_labels = torch.tile(torch.arange(K_plus_1), (n_classes * B,))
        pos_labels = torch.cat([pos_labels, pos_labels]).to(sim_matrix.device)
        
        same_class_mask = (class_labels.unsqueeze(0) == class_labels.unsqueeze(1)).float()
        
        same_pos_mask = (pos_labels.unsqueeze(0) == pos_labels.unsqueeze(1)).float()
        
        strong_positive_mask = (same_class_mask * same_pos_mask).fill_diagonal_(0)
        
        weak_positive_mask = same_class_mask * (1 - same_pos_mask)
        
        negative_mask = 1 - same_class_mask
        
        exp_sim = torch.exp(sim_matrix)
        
        weighted_pos_exp = exp_sim * (strong_positive_mask * 1.0 + weak_positive_mask * 0.1)
        neg_exp = exp_sim * negative_mask
        
        pos_sum = weighted_pos_exp.sum(dim=1)
        neg_sum = neg_exp.sum(dim=1)
        
        losses = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        
        return losses.mean()
    
    def _diversity_regularization(self, labeled_prototypes, unlabeled_prototypes):
       
        diversity_loss = 0
        total_count = 0
        
        for c in range(len(labeled_prototypes)):
            lab_prototypes = labeled_prototypes[c]    # (B, K, D)
            unlab_prototypes = unlabeled_prototypes[c]  # (B, K, D)
            
            B, K, D = lab_prototypes.shape
            
            if K <= 1:
                continue
            
            lab_flat = lab_prototypes.reshape(B * K, D)
            unlab_flat = unlab_prototypes.reshape(B * K, D)
            
            lab_flat_norm = F.normalize(lab_flat, p=2, dim=1, eps=1e-8)
            unlab_flat_norm = F.normalize(unlab_flat, p=2, dim=1, eps=1e-8)
            
            lab_sim = torch.matmul(lab_flat_norm, lab_flat_norm.T)
            unlab_sim = torch.matmul(unlab_flat_norm, unlab_flat_norm.T)
            
            lab_upper_mask = torch.triu(torch.ones_like(lab_sim), diagonal=1) == 1
            lab_upper_sim = lab_sim[lab_upper_mask]
            
            unlab_upper_mask = torch.triu(torch.ones_like(unlab_sim), diagonal=1) == 1
            unlab_upper_sim = unlab_sim[unlab_upper_mask]
            
            lab_penalty = torch.clamp(lab_upper_sim - 0.8, min=0)
            unlab_penalty = torch.clamp(unlab_upper_sim - 0.8, min=0)
            
            diversity_loss += (lab_penalty.mean() + unlab_penalty.mean())
            total_count += 2
        
        return diversity_loss / max(total_count, 1)