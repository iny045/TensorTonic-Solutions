import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.array(v)
    w = np.array(w)
    
    if np.linalg.norm(v) == 0 or np.linalg.norm(w)==0:
        return np.nan
        
    prod = np.dot(v,w)
    magv = np.linalg.norm(v)
    magw = np.linalg.norm(w)

    theta = prod/(magv*magw)
    theta = np.clip(theta, -1.0, 1.0)
    
    return np.arccos(theta)
   