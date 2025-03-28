import numpy as np
import cv2
import os
import scipy.io

def compute_transmission(I, J, A):
    """
    Computes the transmission map T(x) using the atmospheric scattering model.
    I - Hazy image (numpy array)
    J - Haze-free image (numpy array)
    A - Global atmospheric light (scalar or array)
    """
    epsilon = 1e-6  # Avoid division by zero
    T = np.clip((I - A) / (J - A + epsilon), 0.1, 1.0)  # Clip to valid range
    return T

def compute_beta(T, depth_map):
    """
    Computes the scattering coefficient beta using depth maps.
    T - Transmission map (numpy array)
    depth_map - Depth map corresponding to the image (numpy array)
    """
    epsilon = 1e-6  # Avoid division by zero
    valid_pixels = (depth_map > 0)  # Avoid zero depth issues
    beta = -np.log(T[valid_pixels] + epsilon) / (depth_map[valid_pixels] + epsilon)
    return np.mean(beta)  # Take average over the image
'''
def load_depth_map(depth_path):
    """
    Loads a depth map from a .mat file.
    depth_path - Path to the .mat file
    """
    data = scipy.io.loadmat(depth_path)  # Load the .mat file
    # Assuming the depth map variable name is 'depth_map' in the .mat file
    depth_map = data['depth_map']  # Change 'depth_map' to the actual variable name if needed
    # Normalize depth_map to [0, 1] if needed
    depth_map = depth_map.astype(np.float32) / np.max(depth_map)
    return depth_map
'''

def load_depth_map(depth_path):
    """
    Loads a depth map from a .mat file where the matrix is stored without a key.
    """
    data = scipy.io.loadmat(depth_path)  # Load the .mat file
    
    # Remove metadata keys and extract the first available matrix
    valid_keys = [key for key in data.keys() if not key.startswith("__")]  
    if len(valid_keys) == 1:
        depth_map = data[valid_keys[0]]  # Extract the depth matrix
    else:
        raise ValueError(f"Unexpected structure in {depth_path}. Found keys: {list(data.keys())}")

    # Normalize depth_map to [0, 1] if needed
    depth_map = depth_map.astype(np.float32)
    if np.max(depth_map) > 0:
        depth_map /= np.max(depth_map)  # Normalize to [0, 1]

    return depth_map

def process_dataset(hazy_folder, gt_folder, depth_folder, output_file):
    """
    Processes a dataset of images to compute beta values for each pair.
    hazy_folder - Directory with hazy images
    gt_folder - Directory with corresponding ground truth (haze-free) images
    depth_folder - Directory with corresponding depth maps (.mat files)
    output_file - File to save computed beta values
    """
    beta_values = []
    gt_files = os.listdir(gt_folder)
    print('gt_files: ', gt_files)

    for gt_filename in gt_files:
        gt_path = os.path.join(gt_folder, gt_filename)
        depth_path = os.path.join(depth_folder, gt_filename.replace('_RGB.jpg', '_depth.mat'))  # .mat extension for depth map
        print('depth path:', depth_path)
        if not os.path.exists(depth_path):
            continue  # Skip if missing depth map for the GT image

        depth_map = load_depth_map(depth_path)  # Load depth map for the GT image

        # Get the base name (without suffix) to find corresponding hazy images
        base_name = os.path.splitext(gt_filename)[0][:-4]
        print('base_name: ', base_name)
        hazy_files = [f for f in os.listdir(hazy_folder) if f.startswith(base_name)]
        print('hazy_files: ', hazy_files)
        
        if not hazy_files:
            print('hazy_files: ', hazy_files)
            continue  # Skip if no corresponding hazy images found

        # Process each hazy image corresponding to the GT image
        for hazy_filename in hazy_files:
            hazy_path = os.path.join(hazy_folder, hazy_filename)
            I = cv2.imread(hazy_path, cv2.IMREAD_COLOR) / 255.0
            J = cv2.imread(gt_path, cv2.IMREAD_COLOR) / 255.0

            # Ensure all images are the same size
            if I.shape != J.shape:
                J = cv2.resize(J, (I.shape[1], I.shape[0]))
            if I.shape != depth_map.shape:
                depth_map = cv2.resize(depth_map, (I.shape[1], I.shape[0]))

            A = np.max(I)  # Estimate atmospheric light
            T = compute_transmission(I, J, A)
            T = np.mean(T, axis=-1)  # Convert RGB transmission map to grayscale
            beta = compute_beta(T, depth_map)

            beta_values.append((hazy_filename, beta))

    # Save results
    with open(output_file, "w") as f:
        for name, beta in beta_values:
            f.write(f"{name},{beta}\n")

    print(f"Processing complete. Results saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    process_dataset("data/simu", "data/img", "data/depth", "beta_values.csv")

