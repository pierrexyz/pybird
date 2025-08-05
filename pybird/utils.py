import numpy as np
from itertools import permutations, combinations_with_replacement


def diff_all(f, x, max_order=4, epsilon=2.e-3):
    """
    Computes all derivatives of a function f: ℝⁿ → ℝᵐ at point x up to `max_order`
    using central finite differences with memoized evaluation reuse.

    Parameters:
        f         : callable, ℝⁿ → ℝᵐ
        x         : 1D array of shape (n,)
        max_order : int, in [0, 4]; highest derivative order to compute
        epsilon   : scalar or array of shape (n,); finite difference step size

    Returns:
        List of derivatives up to `max_order`, where each element is:
            [0] → f(x)                of shape (m,)
            [1] → Jacobian            of shape (m, n)
            [2] → Hessian             of shape (m, n, n)
            [3] → 3rd-order tensor    of shape (m, n, n, n)
            [4] → 4th-order tensor    of shape (m, n, n, n, n)

        For example, if max_order = 2, the return value is:
            [f(x), Jacobian, Hessian]
    """

    x = np.atleast_1d(x).astype(float)
    fx = np.atleast_1d(f(x))
    m, n = fx.size, x.size
    eye = np.eye(n)

    # --- Handle epsilon: relative (scalar) or per-dimension (array) ---
    if np.isscalar(epsilon):
        eps_rel = epsilon
        eps_abs = 3.e-3 
        epsilon = np.maximum(eps_abs, eps_rel * np.abs(x)) 
    else:
        epsilon = np.broadcast_to(epsilon, x.shape)

    # Cache to store function evaluations
    cache = {}
    def d(dx):
        key = tuple((x + dx).round(12))
        if key not in cache:
            cache[key] = np.atleast_1d(f(x + dx))
        return cache[key]

    def s(*dirs):
        return sum(d * eps for d, eps in zip(np.stack(dirs), epsilon))

    results = [fx.copy()] # 0th-order

    if max_order >= 1:
        J = np.stack([(d(s(e)) - d(s(-e))) / (2 * epsilon[i]) for i, e in enumerate(eye)], axis=-1)
        results.append(J)

    if max_order >= 2:
        H = np.zeros((m, n, n))
        for i, j in combinations_with_replacement(range(n), 2):
            ei, ej = eye[i], eye[j]
            val = (
                d(s(ei, ej)) - d(s(ei, -ej)) - d(s(-ei, ej)) + d(s(-ei, -ej))
            ) / (4 * epsilon[i] * epsilon[j])
            H[:, i, j] = val
            if i != j: H[:, j, i] = val
        results.append(H)

    if max_order >= 3:
        T = np.zeros((m, n, n, n))
        for i, j, k in combinations_with_replacement(range(n), 3):
            ei, ej, ek = eye[i], eye[j], eye[k]
            val = (
                d(s(ei, ej, ek)) - d(s(ei, ej, -ek)) - d(s(ei, -ej, ek)) + d(s(ei, -ej, -ek))
              - d(s(-ei, ej, ek)) + d(s(-ei, ej, -ek)) + d(s(-ei, -ej, ek)) - d(s(-ei, -ej, -ek))
            ) / (8 * epsilon[i] * epsilon[j] * epsilon[k])

            for a, b, c in set(permutations((i, j, k))):
                T[:, a, b, c] = val
        results.append(T)

    if max_order >= 4:
        Q = np.zeros((m, n, n, n, n))
        for i, j, k, l in combinations_with_replacement(range(n), 4):
            ei, ej, ek, el = eye[i], eye[j], eye[k], eye[l]
            val = (
                d(s(ei, ej, ek, el)) - d(s(ei, ej, ek, -el))
              - d(s(ei, ej, -ek, el)) + d(s(ei, ej, -ek, -el))
              - d(s(ei, -ej, ek, el)) + d(s(ei, -ej, ek, -el))
              + d(s(ei, -ej, -ek, el)) - d(s(ei, -ej, -ek, -el))
              - d(s(-ei, ej, ek, el)) + d(s(-ei, ej, ek, -el))
              + d(s(-ei, ej, -ek, el)) - d(s(-ei, ej, -ek, -el))
              + d(s(-ei, -ej, ek, el)) - d(s(-ei, -ej, ek, -el))
              - d(s(-ei, -ej, -ek, el)) + d(s(-ei, -ej, -ek, -el))
            ) / (16 * epsilon[i] * epsilon[j] * epsilon[k] * epsilon[l])

            for a, b, c, d_ in set(permutations((i, j, k, l))):
                Q[:, a, b, c, d_] = val
        results.append(Q)

    return results



def reshape_bird(array, L):
    """
    Reshape 1D concatenated array back to the original arrays (P11l, Pctl, Ploopl).

    Parameters:
        - array: The input array (1D for original, higher dimensions for derivatives)
        - L: The object that contains correlator_sky objects and original shapes

    Returns:
        - A list of reshaped tuples (P11l, Pctl, Ploopl) for each correlator_sky
    """
    original_arrays = []
    idx = 0
    
    for i in range(L.nsky):
        # Get the original shapes of the arrays
        shape_P11l = L.correlator_sky[i].bird.P11l.shape
        shape_Pctl = L.correlator_sky[i].bird.Pctl.shape
        shape_Ploopl = L.correlator_sky[i].bird.Ploopl.shape
        shape_Pstl = L.correlator_sky[i].bird.Pstl.shape
        
        size_cosmo = 3 # f, H, DA
        size_P11l = np.prod(shape_P11l)  # Flattened length of P11l
        size_Pctl = np.prod(shape_Pctl)  # Flattened length of Pctl
        size_Ploopl = np.prod(shape_Ploopl)  # Flattened length of Ploopl
        size_Pstl = np.prod(shape_Pstl)  # Flattened length of Pstl

        # Reshape the 1D original array
        f, H, DA = array[idx:idx + size_cosmo]
        idx += size_cosmo
        P11l = array[idx:idx + size_P11l].reshape(shape_P11l)
        idx += size_P11l
        Pctl = array[idx:idx + size_Pctl].reshape(shape_Pctl)
        idx += size_Pctl
        Ploopl = array[idx:idx + size_Ploopl].reshape(shape_Ploopl)
        idx += size_Ploopl
        Pstl = array[idx:idx + size_Pstl].reshape(shape_Pstl)
        idx += size_Pstl
        
        # Store the reshaped arrays
        original_arrays.append((f, H, DA, P11l, Pctl, Ploopl, Pstl))
    
    return original_arrays


def gauss_lobatto(n):
    """
    Compute the Gauss–Lobatto nodes and weights on the interval [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points (must be ≥ 2)
    
    Returns
    -------
    x : ndarray
        Nodes (abscissas) in [-1, 1]
    w : ndarray
        Corresponding weights
    """
    if n < 2:
        raise ValueError("Gauss-Lobatto rule requires at least 2 nodes")

    

    # Endpoints are always -1 and 1
    x = zeros(n)
    x[0], x[-1] = -1.0, 1.0

    # Interior nodes are the roots of the derivative of P_{n-1}(x)
    if n > 2:
        Pn1 = legendre(n - 1)
        dPn1 = polyder(Pn1)
        x[1:-1] = legroots(dPn1)

    # Compute weights
    w = zeros(n)
    w[0] = w[-1] = 2 / (n * (n - 1))
    for i in range(1, n - 1):
        xi = x[i]
        L = legval(xi, Pn1)
        w[i] = 2 / (n * (n - 1) * (L ** 2))

    return x, w

def gauss_lobatto(n):
    if n < 2:
        raise ValueError("Need at least 2 points")
        
    from numpy import array, concatenate
    from numpy.polynomial.legendre import Legendre

    # Interior nodes: roots of the derivative of P_{n-1}(x)
    Pn1 = Legendre.basis(n - 1)
    dPn1 = Pn1.deriv()
    x = concatenate(([-1.], dPn1.roots(), [-1.]))

    w = array([2 / (n * (n - 1) * Pn1(xi)**2) for xi in x])
    
    return x, w