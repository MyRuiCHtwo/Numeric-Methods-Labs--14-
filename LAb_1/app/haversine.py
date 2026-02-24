import numpy as np


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2

    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_elevation_profile(results, n) -> tuple[list[float], list[float]]:
    coords = [(p["latitude"], p["longitude"]) for p in results]

    distances = [0]
    elevations = [results[0]["elevation"]]
    last_valid_idx = 0
    
    for i in range(1, n):
        d = haversine(*coords[last_valid_idx], *coords[i])
        
        # Фільтр: якщо стрибок більше 500м, вважаємо точку помилковою
        if d > 500:
            continue 
            
        distances.append(distances[-1] + d)
        elevations.append(results[i]["elevation"])
        last_valid_idx = i
    
    return distances, elevations
        