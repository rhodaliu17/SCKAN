import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

class Pro(nn.Module):
    def __init__(self, n_fg_regions=3, n_bg_regions=3, fg_mode='kmeans', bg_mode='kmeans', 
                 enable_visualization=True, save_dir='./vis/'):
        super().__init__()
        self.n_fg_regions = n_fg_regions
        self.n_bg_regions = n_bg_regions
        self.fg_mode = fg_mode
        self.bg_mode = bg_mode
        self.enable_visualization = enable_visualization
        self.save_dir = save_dir
    
    def getPrototype(self, fts, mask, case_name='test', num_classes=None):
        B, C, X, Y, Z = fts.shape
        if num_classes is None:
            num_classes = mask.shape[1]

        class_prototypes_list = [[] for _ in range(num_classes)]

        vis_seed_points_list = []
        vis_region_masks_list = []

        for b in range(B):
            for c in range(num_classes):
                if c == 0:  
                    n_regions = self.n_bg_regions
                    mode = self.bg_mode
                else: 
                    n_regions = self.n_fg_regions
                    mode = self.fg_mode
                
                curr_mask = mask[b, c]  # X x Y x Z
                
                seed_points = self._generate_seed_points(curr_mask, n_regions, mode)
                seed_points = self._sort_seed_points_by_x(seed_points)
                region_masks = self._generate_region_masks_kdtree(curr_mask, seed_points)
                
                batch_prototypes = self._extract_region_prototypes(fts[b], region_masks)
               
                if batch_prototypes.shape[0] < n_regions:
                    padding = torch.zeros(n_regions - batch_prototypes.shape[0], C).to(fts.device)
                    batch_prototypes = torch.cat([batch_prototypes, padding], dim=0)
                elif batch_prototypes.shape[0] > n_regions:
                    batch_prototypes = batch_prototypes[:n_regions]
                
                class_prototypes_list[c].append(batch_prototypes)
                
                if b == 0:
                    vis_seed_points_list.append(seed_points)
                    vis_region_masks_list.append(region_masks)

        prototypes = []
        for c in range(num_classes):
            stacked_prototypes = torch.stack(class_prototypes_list[c], dim=0)  # B x n_regions x C
            
            prototypes.append(stacked_prototypes)

        return prototypes
    

    def _sort_seed_points_by_x(self, seed_points):
        sorted_indices = torch.argsort(seed_points[:, 0])
        return seed_points[sorted_indices]
    
    
    
    def _generate_seed_points(self, mask, n_regions, mode):
        valid_coords = torch.nonzero(mask > 0.5).float()  # N x 3
        
        if valid_coords.shape[0] == 0:
            return torch.zeros(n_regions, 3).to(mask.device)
        
        if valid_coords.shape[0] < n_regions:
            repeat_times = (n_regions + valid_coords.shape[0] - 1) // valid_coords.shape[0]
            valid_coords = valid_coords.repeat(repeat_times, 1)
        
        return self._kmeans_sampling(valid_coords, n_regions)
        
   

    def _kmeans_sampling(self, points, n_samples, downsample_factor=2):
        if points.shape[0] < n_samples:
            repeat_times = (n_samples + points.shape[0] - 1) // points.shape[0]
            points = points.repeat(repeat_times, 1)
        
        downsampled_points = points / downsample_factor
        
        downsampled_points_rounded = torch.round(downsampled_points)
        unique_points, inverse_indices = torch.unique(downsampled_points_rounded, dim=0, return_inverse=True)
        if unique_points.shape[0] < n_samples:
            if unique_points.shape[0] == 0:
                return points[:n_samples]
            repeat_times = (n_samples + unique_points.shape[0] - 1) // unique_points.shape[0]
            selected_downsampled = unique_points.repeat(repeat_times, 1)[:n_samples]
        else:
            unique_points_np = unique_points.detach().cpu().numpy()
            kmeans = MiniBatchKMeans(n_clusters=n_samples, random_state=42, 
                                batch_size=min(1000, unique_points_np.shape[0]))
            kmeans.fit(unique_points_np)

            
            centers_np = kmeans.cluster_centers_
            centers = torch.from_numpy(centers_np).float().to(points.device)
            
            distances = torch.cdist(centers, unique_points)
            closest_indices = torch.argmin(distances, dim=1)
            selected_downsampled = unique_points[closest_indices]
        
        selected_upsampled = selected_downsampled * downsample_factor
        distances = torch.cdist(selected_upsampled, points)
        closest_indices = torch.argmin(distances, dim=1)
        selected_points = points[closest_indices]
        
        return selected_points


    def _generate_region_masks_kdtree(self, mask, seed_points):
        X, Y, Z = mask.shape
        n_regions = seed_points.shape[0]
        
        valid_coords = torch.nonzero(mask > 0.5).float()  # N x 3
        
        if valid_coords.shape[0] == 0:
            return torch.zeros(n_regions, X, Y, Z).to(mask.device)
        
        valid_coords_np = valid_coords.cpu().numpy()
        seed_points_np = seed_points.cpu().numpy()
        kdtree = KDTree(seed_points_np)
        distances, closest_seeds = kdtree.query(valid_coords_np, k=1)
        closest_seeds = closest_seeds.flatten() 
        
        closest_seeds_tensor = torch.from_numpy(closest_seeds).to(mask.device)
        
        region_masks = torch.zeros(n_regions, X, Y, Z).to(mask.device)
        
        valid_coords_long = valid_coords.long() 
        x_coords = valid_coords_long[:, 0]
        y_coords = valid_coords_long[:, 1] 
        z_coords = valid_coords_long[:, 2]
        
        for region_id in range(n_regions):
            mask_indices = closest_seeds_tensor == region_id
            if mask_indices.sum() > 0:
                region_x = x_coords[mask_indices]
                region_y = y_coords[mask_indices]
                region_z = z_coords[mask_indices]
                region_masks[region_id, region_x, region_y, region_z] = 1.0
        
        return region_masks
    
    def _extract_region_prototypes(self, fts, region_masks):
        C = fts.shape[0]
        n_regions = region_masks.shape[0]
        prototypes = torch.zeros(n_regions, C).to(fts.device)
        
        for r in range(n_regions):
            region_mask = region_masks[r]  # X x Y x Z
            if region_mask.sum() > 0:
                masked_fts = fts * region_mask[None, ...]  # C x X x Y x Z
                region_sum = torch.sum(masked_fts, dim=(1, 2, 3))  # C
                region_count = region_mask.sum()  # scalar
                prototypes[r] = region_sum / region_count
            else:
                prototypes[r] = torch.zeros(C).to(fts.device)
        
        return prototypes
    
    
    def calDist(self, fts, prototypes, eps=1e-8):
        N, C, X, Y, Z = fts.shape
        K = prototypes.size(0)
        
        fts_reshaped = fts.view(N, C, -1)  # [N, C, X*Y*Z]
        prototypes_reshaped = prototypes.view(K, C)  # [K, C]
        
        numerator = torch.einsum('nci,kc->nki', fts_reshaped, prototypes_reshaped)
        
        fts_norm = torch.norm(fts_reshaped, dim=1, keepdim=True)  # [N, 1,X*Y*Z]
        proto_norm = torch.norm(prototypes_reshaped, dim=1, keepdim=True)  # [K,1]
        
        norm_product = fts_norm * proto_norm.unsqueeze(0)  # [N, K, X*Y*Z]
        
        denominator = torch.clamp(norm_product, min=eps)  # (norm_product, eps)
        
        similarity = numerator / denominator  # [N, K, X*Y*Z]
        
        max_similarity, _ = torch.max(similarity, dim=1)  # [N, X*Y*Z]
        
        return max_similarity.view(N, X, Y, Z)