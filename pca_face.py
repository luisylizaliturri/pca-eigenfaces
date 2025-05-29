from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename).astype(float)
    mean = np.mean(x, axis=0)
    x_centered = x - mean
    return x_centered

def get_covariance(dataset):
    n = len(dataset)
    # Compute covariance matrix
    S = np.dot(np.transpose(dataset), dataset) / (n - 1)
    return S

def get_eig(S, k):
    length = len(S)

    #Calculate k largest eigenvalues and corresponding normalized eignevectors
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[length-k, length-1])
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Crate k * k diagonal matrix
    Lambda = np.diag(eigenvalues)
    return Lambda, eigenvectors

def get_eig_prop(S, prop):
    eigenvalues, eigenvectors = eigh(S)

    #Sort descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    total_variance = np.sum(sorted_eigenvalues)
    selection = sorted_eigenvalues / total_variance > prop
    if not np.any(selection):
        return np.array([]), np.array([[]])
    selected_eigenvalues = sorted_eigenvalues[selection]
    selected_eigenvectors = sorted_eigenvectors[:, selection]
    Lambda = np.diag(selected_eigenvalues)

    return Lambda, selected_eigenvectors


def project_image(image, U):
    image = image.reshape(-1, 1)
    score = np.dot(np.transpose(U), image) 
    x_pca = np.dot(U, score)
    x_pca = x_pca.flatten()
    return x_pca

def display_image(orig, proj):
    orig_image = orig.reshape(64, 64)
    proj_image = proj.reshape(64, 64)

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)

    #Plot original image
    im1 = ax1.imshow(orig_image, aspect='equal')
    ax1.set_title("Original")
    plt.colorbar(im1, ax=ax1)

    #Plot reconstructed image
    im2 = ax2.imshow(proj_image, aspect='equal')
    ax2.set_title("Projection")
    plt.colorbar(im2, ax=ax2)

    return fig, ax1, ax2

def perturb_image(image, U, sigma):
    image = image.reshape(-1, 1)

    #Original pca projection weights
    score = np.dot(np.transpose(U), image) 

    #Generate Gaussian noise
    noise = np.random.normal(0, sigma, size=score.shape)
    perturbed_score = score + noise  

    #Reconstruct perturbed image
    x_perturbed = np.dot(U, perturbed_score) 

    x_perturbed = x_perturbed.flatten()
    return x_perturbed

def combine_image(image1, image2, U, lam):
    image1 = image1.reshape(-1, 1)  
    image2 = image2.reshape(-1, 1) 

    #Compute pca projection weights
    score1 = np.dot(np.transpose(U), image1)  
    score2 = np.dot(np.transpose(U), image2) 

    #Compute convex combination
    score_comb = lam * score1 + (1 - lam) * score2

    #Reconstruct combined image
    x_comb = np.dot(U, score_comb) 
    x_comb = x_comb.flatten()

    return x_comb
