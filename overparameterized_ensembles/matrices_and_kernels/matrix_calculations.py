import torch
from overparameterized_ensembles.utils.constants import (
    ZERO_REGULARIZATION,
)


def cholesky_decomposition(K):
    """
    Computes the Cholesky decomposition of a positive definite matrix K.

    Parameters
    ----------
    K : torch.Tensor
        A positive definite matrix.

    Returns
    -------
    torch.Tensor
        The Cholesky decomposition of K, i.e. an upper triangular matrix R such that K = R^T R.
    """
    K_stable = K + ZERO_REGULARIZATION * torch.eye(K.size(0))

    return torch.linalg.cholesky(K_stable, upper=True)
