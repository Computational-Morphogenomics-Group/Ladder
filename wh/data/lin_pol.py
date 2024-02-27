########################################
# Code used to generate synthetic data
# Auth: @zgr2788 
# TODO: Add more


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
########################################



def generate_simulated_data(d, t, n, p_deg=0, c_s=1, a_s=1, idx_s=1, e_m=0, e_s=1, linear=True):
    """
    Parameters
    ----------
    d : int
        number of features for generated points
    t : int
        number of time points to generate
    n : int
        number of points to generate
    p_deg : int, optional
        degree of the polynomial to be used, has no effect if ``linear==True``
    c_s : float, optional
        factor used to scale coefficients
    a_s : float, optional
        factor used to scale idx_s, e_m and e_s together
    idx_s : float, optional
        factor used to scale time indices when generating, higher values indicate greater separation along timepoints
    e_m : float,optional
        error mean
    e_s : float, optional
        error std

        
    Returns
    -------
    data : array-like, shape (n, t, d)
        generated points across time
    poly_coeffs : array-like, shape (n, d-1)
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import syngen
    >>> syngen.generate_simulated_data(2,2,2)
    (array([[[ 0.18126027,  0.65219774],
        [ 1.30571011, -0.55590506]],
       [[-0.50006976, -1.141606  ],
        [ 1.46194153,  0.54011249]]]), array([[0.29394387],
       [0.72274433]]))
    
    """
    data = np.zeros((n, t, d))

    if linear:
        poly_coeffs = np.random.normal(0,1,size=(d - 1)) * c_s

    else:
        poly_coeffs = np.random.normal(0,1,size=(d-1, p_deg)) * c_s


    
    for i in range(n):
        index = np.arange(t).reshape(-1, 1) * idx_s * a_s
        
        if linear:
            poly_terms = index.dot(poly_coeffs.reshape(1, -1))

            
        else:
            idx_flat = index.flatten()
            poly_terms = np.array([[np.polyval(poly_coeffs[j], idx_flat[i]) for j in range(d-1)]  for i in range(t)])



        noise = np.random.normal(e_m * a_s, e_s * np.sqrt(a_s), (t, d))
        
        data[i, :, :] = np.concatenate([index, poly_terms], axis=1) + noise
            
    return data, poly_coeffs
