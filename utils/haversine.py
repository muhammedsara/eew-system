"""
Haversine Distance Calculations for Spatial Clustering

Used in DBSCAN spatial clustering for grouping earthquake detections
within ε=5km radius (Paper Section III.B, Stage 1)
"""

import numpy as np
from typing import Union, List, Tuple

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Uses the Haversine formula for accurate spherical distance calculation.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
        
    Example:
        >>> haversine_distance(40.5, 30.2, 40.6, 30.3)
        13.47  # approximately
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def haversine_distance_matrix(
    coordinates: np.ndarray
) -> np.ndarray:
    """
    Calculate pairwise Haversine distance matrix for DBSCAN clustering.
    
    Args:
        coordinates: Array of shape (n, 2) with [lat, lon] pairs
        
    Returns:
        Distance matrix of shape (n, n) in kilometers
        
    Note:
        This is used by DBSCAN for spatial clustering with eps=5km
    """
    n = len(coordinates)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(
                coordinates[i, 0], coordinates[i, 1],
                coordinates[j, 0], coordinates[j, 1]
            )
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def haversine_distance_vectorized(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance for batch calculations.
    
    Args:
        lat1, lon1: Arrays of latitudes and longitudes for first points
        lat2, lon2: Arrays of latitudes and longitudes for second points
        
    Returns:
        Array of distances in kilometers
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def points_within_radius(
    center_lat: float, center_lon: float,
    points: np.ndarray,
    radius_km: float
) -> np.ndarray:
    """
    Find all points within a given radius of a center point.
    
    Used for determining which devices are affected by an earthquake.
    
    Args:
        center_lat, center_lon: Epicenter coordinates
        points: Array of shape (n, 2) with [lat, lon] pairs
        radius_km: Affected radius in kilometers
        
    Returns:
        Boolean mask of points within radius
    """
    distances = haversine_distance_vectorized(
        np.full(len(points), center_lat),
        np.full(len(points), center_lon),
        points[:, 0],
        points[:, 1]
    )
    
    return distances <= radius_km


def calculate_affected_radius(magnitude: float) -> float:
    """
    Calculate affected radius based on earthquake magnitude.
    
    Formula from paper: r = 10^(M-3) km
    
    Args:
        magnitude: Earthquake magnitude (Richter scale)
        
    Returns:
        Affected radius in kilometers
        
    Examples:
        M4.5 → 31.6 km
        M5.5 → 316 km
        M6.5 → 3162 km
    """
    return 10 ** (magnitude - 3)


def estimate_epicenter(
    coordinates: np.ndarray,
    weights: np.ndarray = None
) -> Tuple[float, float]:
    """
    Estimate earthquake epicenter from triggered device locations.
    
    Uses weighted centroid of triggered devices.
    
    Args:
        coordinates: Array of shape (n, 2) with [lat, lon] pairs
        weights: Optional weights for each coordinate (e.g., confidence scores)
        
    Returns:
        Estimated epicenter (lat, lon)
    """
    if weights is None:
        weights = np.ones(len(coordinates))
    
    weights = weights / weights.sum()
    
    # Weighted centroid
    lat = np.sum(coordinates[:, 0] * weights)
    lon = np.sum(coordinates[:, 1] * weights)
    
    return lat, lon
