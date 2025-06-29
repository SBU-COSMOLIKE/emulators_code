import numpy as np

def composite_simpson(f, a, b, n):
    """
    Composite Simpson's rule over [a, b] with n even subintervals.
    """
    if n % 2:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return S * h / 3
def romberg_simpson(f, a, b, tol=1e-8, max_levels=6):
    """
    Romberg integration using composite Simpson's rule as the base sequence.
    
    Builds R[i][0] = S(h_i) with h_i = (b-a)/(2^i * base_n),
    then Richardson-extrapolates:
      R[i][j] = (16^j * R[i][j-1] - R[i-1][j-1]) / (16^j - 1)
    until |R[i][i] - R[i-1][i-1]| < tol or max_levels reached.
    
    Returns:
      - R[i][i]: the extrapolated integral estimate
      - R: full Romberg table (list of lists)
    """
    # initialize Romberg table
    R = [[0.0]*(max_levels+1) for _ in range(max_levels+1)]
    
    # base number of Simpson intervals
    base_n = 2
    for i in range(max_levels+1):
        n_i = base_n * 2**i  # ensure even
        R[i][0] = composite_simpson(f, a, b, n_i)
        
        # Richardson extrapolation
        for j in range(1, i+1):
            factor = 16**j
            R[i][j] = (factor * R[i][j-1] - R[i-1][j-1]) / (factor - 1)
        
        # convergence check
        if i > 0 and abs(R[i][i] - R[i-1][i-1]) < tol:
            return R[i][i], R
    
    return R[max_levels][max_levels], R