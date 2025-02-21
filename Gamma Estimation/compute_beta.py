import numpy as np
import cv2
import os

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

def process_dataset(hazy_folder, gt_folder, depth_folder, output_file):
    """
    Processes a dataset of images to compute beta values for each pair.
    hazy_folder - Directory with hazy images
    gt_folder - Directory with corresponding ground truth (haze-free) images
    depth_folder - Directory with corresponding depth maps
    output_file - File to save computed beta values
    """
    beta_values = []
    files = os.listdir(hazy_folder)

    for filename in files:
        hazy_path = os.path.join(hazy_folder, filename)
        gt_path = os.path.join(gt_folder, filename)
        depth_path = os.path.join(depth_folder, filename)

        if not os.path.exists(gt_path) or not os.path.exists(depth_path):
            continue  # Skip if missing data

        I = cv2.imread(hazy_path, cv2.IMREAD_COLOR) / 255.0
        J = cv2.imread(gt_path, cv2.IMREAD_COLOR) / 255.0
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 255.0

        A = np.max(I)  # Estimate atmospheric light
        T = compute_transmission(I, J, A)
        beta = compute_beta(T, depth_map)

        beta_values.append((filename, beta))

    # Save results
    with open(output_file, "w") as f:
        for name, beta in beta_values:
            f.write(f"{name},{beta}\n")

# Example usage
if __name__ == "__main__":
    process_dataset("data/hazy", "data/gt", "data/depth", "beta_values.csv")
