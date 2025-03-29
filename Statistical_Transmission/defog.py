import numpy as np

def defog(HazeImg, t, A, delta):
    # Ensure t is above a minimum threshold before applying delta exponentiation.
    t = np.maximum(np.abs(t), 1e-5) ** delta
    t = np.clip(t ** delta, 0.2, 1.0)  # Avoid very small values

    # If A is a single value, replicate it for three channels.
    if np.isscalar(A) or (isinstance(A, np.ndarray) and A.size == 1):
        A = np.full((3,), A)
    else:
        A = np.array(A).flatten()
    
    # Process each channel: subtract A, divide by t, then add A back.
    R = (HazeImg[..., 0] - A[0]) / t + A[0]
    G = (HazeImg[..., 1] - A[1]) / t + A[1]
    B = (HazeImg[..., 2] - A[2]) / t + A[2]
    print("before clipping r -> ", np.max(R), np.min(R))
    
    # Clamp each channel between 0 and 1.
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    print("after clipping R -> ", np.max(R), np.min(R))
    print("after clipping G -> ", np.max(G), np.min(G))
    print("after clipping B -> ", np.max(B), np.min(B))
    
    print("B -> ",B)
    
    # Stack the channels back into a 3D image.
    rImg = np.stack((R, G, B), axis=-1)
    return rImg

