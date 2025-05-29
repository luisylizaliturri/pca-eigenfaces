# usage_examples.py

"""
Usage Examples for pca_face.py

This script demonstrates how to use the functions implemented in pca_face.py to perform PCA-based facial analysis.
Ensure that pca_face.py is in the same directory as this script or is accessible via Python's import path.

Author: Luis Ylizaliturri
Date: 5/28/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pca_face import (
    load_and_center_dataset,
    get_covariance,
    get_eig,
    get_eig_prop,
    project_image,
    display_image,
    perturb_image,
    combine_image
)

def main():
    # 1. Load and Center the Dataset
    print("Loading and centering the dataset...")
    filename = 'face_dataset.npy'  # Ensure this file is in the same directory or provide the correct path
    try:
        x_centered = load_and_center_dataset(filename)
        print(f"Dataset loaded. Number of images: {len(x_centered)}")
        print(f"Number of features per image: {len(x_centered[0])}")
        print(f"Average value of centered dataset: {np.average(x_centered)}")
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found. Please check the path and try again.")
        return

    # 2. Compute the Covariance Matrix
    print("\nComputing the covariance matrix...")
    S = get_covariance(x_centered)
    print(f"Covariance matrix shape: {S.shape}")

    # 3. Perform Eigendecomposition to Get Top k Eigenvalues/Eigenvectors
    k = 100  # You can choose any k based on your requirements
    print(f"\nPerforming eigendecomposition to get the top {k} eigenvalues and eigenvectors...")
    Lambda, U = get_eig(S, k)
    print(f"Top {k} eigenvalues:\n{Lambda}")

    # 4. Perform Eigendecomposition to Get Eigenvalues/Eigenvectors Explaining a Proportion of Variance
    prop = 0.95  # For example, 95% of the variance
    print(f"\nPerforming eigendecomposition to get eigenvalues/eigenvectors explaining more than {prop*100}% of the variance...")
    Lambda_prop, U_prop = get_eig_prop(S, prop)
    print(f"Number of eigenvalues/eigenvectors selected: {Lambda_prop.shape[0]}")
    print(f"Selected eigenvalues:\n{Lambda_prop}")

    # 5. Project and Reconstruct an Image
    image_index = 40  # You can choose any valid index
    print(f"\nProjecting and reconstructing image at index {image_index} using top {k} eigenvectors...")
    original_image = x_centered[image_index]
    reconstructed_image = project_image(original_image, U)
    fig, ax1, ax2 = display_image(original_image, reconstructed_image)
    plt.suptitle(f"Image Projection using Top {k} Eigenvectors")
    plt.show()

    # 6. Perturb the Projection Weights and Reconstruct the Perturbed Image
    sigma = 1000  # Standard deviation for Gaussian noise
    print(f"\nPerturbing the projection weights of image at index {image_index} with sigma = {sigma}...")
    perturbed_image = perturb_image(original_image, U, sigma)
    fig, ax1, ax2 = display_image(original_image, perturbed_image)
    plt.suptitle(f"Perturbed Projection (sigma={sigma})")
    plt.show()

    # 7. Create a Convex Combination of Two Images
    image_index_1 = 50
    image_index_2 = 70
    lam = 0.5  # Convex combination parameter (0 <= lam <= 1)
    print(f"\nCreating a convex combination of images at indices {image_index_1} and {image_index_2} with lambda = {lam}...")
    image1 = x_centered[image_index_1]
    image2 = x_centered[image_index_2]
    combined_image = combine_image(image1, image2, U, lam)
    fig, ax1, ax2 = display_image(image2, combined_image)
    plt.suptitle(f"Convex Combination of Images {image_index_1} and {image_index_2} (lambda={lam})")
    plt.show()

    # 8. Optional: Visualize Eigenfaces
    visualize_eigenfaces = True
    if visualize_eigenfaces:
        print("\nVisualizing the first 5 eigenfaces...")
        num_eigenfaces = 5
        eigenfaces = U[:, :num_eigenfaces]
        fig, axes = plt.subplots(1, num_eigenfaces, figsize=(15, 3))
        for i in range(num_eigenfaces):
            eigenface = eigenfaces[:, i].reshape(64, 64)
            axes[i].imshow(eigenface)
            axes[i].set_title(f"Eigenface {i+1}")
            axes[i].axis('off')
        plt.suptitle("First 5 Eigenfaces")
        plt.show()

# def test3(x,U):
#     print("Test 3\n")
#     combined_image = combine_image(x[50], x[80], U, 0.5)
#     fig, ax1, ax2 = display_image(x[80], combined_image) # this function is similar t
#     plt.show()


# def test2(x, U):
#     print("Test 2\n")
#     perturbed_image = perturb_image(x[50], U, 1000)
#     fig, ax1, ax2 = display_image(x[50], perturbed_image)
#     plt.show()

# def test1(x, U):
#     print("Test 1\n")
#     projection = project_image(x[50], U)
#     fig, ax1, ax2 = display_image(x[50], projection)
#     plt.show()

# def main():
#     print("Main")
#     x = load_and_center_dataset('face_dataset.npy')
#     S = get_covariance(x)
#     Lambda, U = get_eig(S, 100)
    
#     test1(x, U)
#     test2(x, U)
#     test3(x, U)

if __name__ == "__main__":
    main()
