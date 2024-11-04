import torch
from overparameterized_ensembles.models.model_utils import (
    sample_omega_and_calculate_phi,
)
from overparameterized_ensembles.matrices_and_kernels.matrix_calculations import (
    cholesky_decomposition,
)
from overparameterized_ensembles.utils.constants import (
    ZERO_REGULARIZATION,
)


def calculate_w_term(
    X_star, K_star, num_features, random_weights_distribution, activation_function_name
):
    # Calculate the Phi matrix (including the test point)
    Phi_star = sample_omega_and_calculate_phi(
        X_star, activation_function_name, random_weights_distribution, num_features
    )

    # Calculate W and w_perp
    W, w_perp = calculate_W_and_w_perp(K_star, Phi_star)

    # Calculate W * w_perp
    W_w_perp = torch.mm(W, w_perp.unsqueeze(1))

    # Solve (W * W^T) x = (W * w_perp^T) for x using lstsq and a small ridge term for numerical stability (and comparability with the other experiments)
    lhs = W.mm(W.t()) + ZERO_REGULARIZATION * torch.eye(W.size(0))
    x = torch.linalg.lstsq(lhs, W_w_perp, driver="gelsd").solution

    return x.t()


def calculate_w_term_ridge(
    X_star,
    K_star,
    num_features,
    random_weights_distribution,
    activation_function_name,
    ridge,
):
    # Calculate the Phi matrix (including the test point)
    Phi_star = sample_omega_and_calculate_phi(
        X_star, activation_function_name, random_weights_distribution, num_features
    )

    # Calculate W and w_perp
    W, w_perp = calculate_W_and_w_perp(K_star, Phi_star)

    # Calculate W * w_perp
    W_w_perp = torch.mm(W, w_perp.unsqueeze(1))

    # Calculate the Cholesky decomposition of the kernel matrix
    # Perform the Cholesky decomposition
    R_star = cholesky_decomposition(K_star)

    # Extract R without the last row and column for W calculation
    R_upper = R_star[:-1, :-1]

    # Solve (W * W^T + ridge * D * R^{-\top} R^{-1}) x = (W * w_perp^T) for x using lstsq and a small ridge term for numerical stability (and comparability with the other experiments)
    lhs = W.mm(W.t()) + ridge * W.size(1) * torch.linalg.solve_triangular(
        R_upper.t(),
        torch.linalg.solve_triangular(R_upper, torch.eye(R_upper.size(0)), upper=False),
        upper=False,
    )
    x = torch.linalg.lstsq(lhs, W_w_perp, driver="gelsd").solution

    return x.t()


def calculate_W(R, Phi):
    """
    Calculate the matrix W for a given R matrix and Phi matrix.

    Args:
    R : torch.Tensor
        The R matrix of shape (n_samples, n_samples) and is the Cholesky decomposition of the kernel matrix. It especially is an upper triangular matrix.
    Phi : torch.Tensor
        The Phi matrix of shape (n_samples, num_features)

    Returns:
    W : torch.Tensor
        The matrix W of shape (n_samples, num_features)
    """
    # Compute the matrix W
    W = torch.linalg.solve_triangular(R.t(), Phi, upper=False)

    return W


def calculate_w_perp(W, phi_star, c, r_perp):
    """
    Calculate the vector w_perp for a given W matrix, phi_star vector, c vector, and r_perp scalar.

    Args:
    W : torch.Tensor
        The W matrix of shape (n_samples, num_features)
    phi_star : torch.Tensor
        The phi_star vector of shape (num_features,)
    c : torch.Tensor
        The c vector of shape (n_samples,)
    r_perp : float
        The r_perp scalar

    Returns:
    w_perp : torch.Tensor
        The w_perp vector of shape (num_features,)
    """

    W_top_c = torch.mm(W.t(), c.unsqueeze(1)).squeeze()

    w_perp = (phi_star.t() - W_top_c) / r_perp

    return w_perp


def calculate_W_and_w_perp(K_star, Phi_star):
    """
    Calculate matrices W and w_perp using the (N+1)x(N+1) Gram matrix K_star.

    Args:
    K_star : torch.Tensor
        The (N+1)x(N+1) kernel matrix.
    Phi_star : torch.Tensor
        The Phi matrix of shape (N+1, num_features) that includes phi* as the last row.
    tau : torch.distributions.Distribution
        The distribution from which omega parameters are sampled.
    num_features : int
        The number of random features.

    Returns:
    W : torch.Tensor
        The matrix W of shape (N, num_features).
    w_perp : torch.Tensor
        The vector w_perp of shape (num_features,).
    """
    # Perform the Cholesky decomposition
    R_star = cholesky_decomposition(K_star)

    # Extract R without the last row and column for W calculation
    R_upper = R_star[:-1, :-1]

    # Extract the last column of R, which contains [c, r_perp]
    R_last = R_star[:, -1]

    # Extract Phi without the last row and phi_star which is the last row
    Phi = Phi_star[:-1]
    phi_star = Phi_star[-1]

    # Calculate W using the upper part of R (ignoring the last row and column)
    W = calculate_W(R_upper, Phi)

    # Extract c and r_perp from the last row of R
    c = R_last[:-1]
    r_perp = R_last[-1]

    # Calculate w_perp using phi_star, W, c, and r_perp
    w_perp = calculate_w_perp(W, phi_star, c, r_perp)

    return W, w_perp
