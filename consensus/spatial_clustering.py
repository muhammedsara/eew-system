"""
Spatial Clustering (DBSCAN with Haversine Distance)

Stage 1 of the consensus protocol (Paper Section III.B):
"We apply DBSCAN with Haversine distance metric to group detections 
within ε=5km radius. This radius ensures devices triggered by the 
same seismic wavefront (S-wave velocity ≈3 km/s) cluster together 
within our temporal window."

Author: Muhammed Şara
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils.haversine import haversine_distance, haversine_distance_matrix


@dataclass
class SpatialCluster:
    """Represents a spatial cluster of device triggers"""
    cluster_id: int
    triggers: List[Dict]
    center_lat: float
    center_lon: float
    radius_km: float
    mobile_count: int
    iot_count: int
    
    @property
    def total_count(self) -> int:
        return self.mobile_count + self.iot_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'center': {'lat': self.center_lat, 'lon': self.center_lon},
            'radius_km': self.radius_km,
            'mobile_count': self.mobile_count,
            'iot_count': self.iot_count,
            'total_count': self.total_count,
            'triggers': self.triggers
        }


class SpatialClusterer:
    """
    DBSCAN-based spatial clustering for earthquake trigger grouping.
    
    Uses Haversine distance to properly handle geographic coordinates.
    Groups triggers that are within ε=5km of each other.
    
    Attributes:
        eps_km: Epsilon radius in kilometers (default: 5km)
        min_samples: Minimum triggers to form a cluster (default: 3)
    """
    
    def __init__(
        self,
        eps_km: float = None,
        min_samples: int = None
    ):
        """
        Initialize spatial clusterer.
        
        Args:
            eps_km: Epsilon radius (default from config: 5km)
            min_samples: Minimum samples per cluster (default: 3)
        """
        self.eps_km = eps_km or config.consensus.spatial_eps_km
        self.min_samples = min_samples or config.consensus.min_samples
    
    def cluster(self, triggers: List[Dict]) -> List[SpatialCluster]:
        """
        Cluster triggers by spatial proximity.
        
        Args:
            triggers: List of trigger dictionaries with 'lat', 'lon' keys
            
        Returns:
            List of SpatialCluster objects
        """
        if len(triggers) < self.min_samples:
            return []
        
        # Extract coordinates
        coords = np.array([
            [t.get('lat', t.get('latitude', 0)),
             t.get('lon', t.get('longitude', 0))]
            for t in triggers
        ])
        
        # Calculate distance matrix using Haversine
        distance_matrix = haversine_distance_matrix(coords)
        
        # Run DBSCAN with precomputed distances
        clustering = DBSCAN(
            eps=self.eps_km,
            min_samples=self.min_samples,
            metric='precomputed'
        ).fit(distance_matrix)
        
        labels = clustering.labels_
        
        # Group triggers by cluster
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get triggers in this cluster
            cluster_mask = labels == label
            cluster_triggers = [t for t, m in zip(triggers, cluster_mask) if m]
            cluster_coords = coords[cluster_mask]
            
            # Calculate cluster center (centroid)
            center_lat = np.mean(cluster_coords[:, 0])
            center_lon = np.mean(cluster_coords[:, 1])
            
            # Calculate cluster radius
            distances = np.array([
                haversine_distance(center_lat, center_lon, c[0], c[1])
                for c in cluster_coords
            ])
            radius = np.max(distances) if len(distances) > 0 else 0
            
            # Count device types
            mobile_count = sum(
                1 for t in cluster_triggers 
                if t.get('device_type') == 'mobile'
            )
            iot_count = sum(
                1 for t in cluster_triggers 
                if t.get('device_type') == 'iot_anchor'
            )
            
            clusters.append(SpatialCluster(
                cluster_id=int(label),
                triggers=cluster_triggers,
                center_lat=float(center_lat),
                center_lon=float(center_lon),
                radius_km=float(radius),
                mobile_count=mobile_count,
                iot_count=iot_count
            ))
        
        return clusters
    
    def cluster_incremental(
        self,
        new_trigger: Dict,
        existing_clusters: List[SpatialCluster]
    ) -> Tuple[Optional[int], List[SpatialCluster]]:
        """
        Incrementally add a trigger to existing clusters.
        
        For real-time processing, assigns new triggers to nearest cluster
        or creates a new cluster point.
        
        Args:
            new_trigger: New trigger dictionary
            existing_clusters: Current list of clusters
            
        Returns:
            Tuple of (assigned_cluster_id, updated_clusters)
        """
        new_lat = new_trigger.get('lat', new_trigger.get('latitude', 0))
        new_lon = new_trigger.get('lon', new_trigger.get('longitude', 0))
        
        # Find nearest cluster
        min_dist = float('inf')
        nearest_cluster_idx = None
        
        for i, cluster in enumerate(existing_clusters):
            dist = haversine_distance(
                new_lat, new_lon,
                cluster.center_lat, cluster.center_lon
            )
            if dist < min_dist:
                min_dist = dist
                nearest_cluster_idx = i
        
        # Check if within epsilon
        if min_dist <= self.eps_km and nearest_cluster_idx is not None:
            # Add to existing cluster
            cluster = existing_clusters[nearest_cluster_idx]
            cluster.triggers.append(new_trigger)
            
            # Update counts
            if new_trigger.get('device_type') == 'mobile':
                cluster.mobile_count += 1
            else:
                cluster.iot_count += 1
            
            # Update center (running average)
            n = len(cluster.triggers)
            cluster.center_lat = (
                cluster.center_lat * (n - 1) + new_lat
            ) / n
            cluster.center_lon = (
                cluster.center_lon * (n - 1) + new_lon
            ) / n
            
            return cluster.cluster_id, existing_clusters
        else:
            # Create new cluster (as single point)
            # Will be promoted to full cluster when enough points
            return None, existing_clusters
    
    def merge_overlapping_clusters(
        self,
        clusters: List[SpatialCluster]
    ) -> List[SpatialCluster]:
        """
        Merge clusters that have overlapping coverage.
        
        Args:
            clusters: List of spatial clusters
            
        Returns:
            Merged list of clusters
        """
        if len(clusters) <= 1:
            return clusters
        
        # Calculate pairwise distances between cluster centers
        merged = []
        used = set()
        
        for i, c1 in enumerate(clusters):
            if i in used:
                continue
            
            # Find clusters to merge
            to_merge = [c1]
            used.add(i)
            
            for j, c2 in enumerate(clusters):
                if j in used or j <= i:
                    continue
                
                dist = haversine_distance(
                    c1.center_lat, c1.center_lon,
                    c2.center_lat, c2.center_lon
                )
                
                # Merge if centers are within 2*epsilon
                if dist <= 2 * self.eps_km:
                    to_merge.append(c2)
                    used.add(j)
            
            # Create merged cluster
            if len(to_merge) == 1:
                merged.append(c1)
            else:
                all_triggers = []
                all_mobile = 0
                all_iot = 0
                
                for c in to_merge:
                    all_triggers.extend(c.triggers)
                    all_mobile += c.mobile_count
                    all_iot += c.iot_count
                
                # Calculate new center
                lats = [t.get('lat', t.get('latitude', 0)) for t in all_triggers]
                lons = [t.get('lon', t.get('longitude', 0)) for t in all_triggers]
                
                merged_cluster = SpatialCluster(
                    cluster_id=c1.cluster_id,
                    triggers=all_triggers,
                    center_lat=np.mean(lats),
                    center_lon=np.mean(lons),
                    radius_km=self.eps_km,
                    mobile_count=all_mobile,
                    iot_count=all_iot
                )
                merged.append(merged_cluster)
        
        return merged
