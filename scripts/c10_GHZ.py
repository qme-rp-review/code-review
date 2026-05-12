import numpy as np

DTYPE = np.complex64

_T = np.array([[0,1,1,0],
               [0,1j,-1j,0],
               [1,0,0,-1],
               [1,0,0,1]], dtype=DTYPE)

def _infer_r_from_len(d: int) -> int:
    # d = 4^r = 2^(2r)
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError("vecR length must be a power of 2 (actually 4^r).")
    e = d.bit_length() - 1
    if e % 2 != 0:
        raise ValueError("vecR length must be 4^r (i.e., 2^(even)).")
    return e // 2

def precompute_Wt_from_vecR(vecR: np.ndarray) -> np.ndarray:
    """
    vecR (length 4^r, Pauli-basis coefficients) -> W^T (shape 2^r x 2^r) in computational basis,
    consistent with your state_vec_Pauli definition.
    """
    vecR = np.asarray(vecR, dtype=DTYPE)
    d = vecR.size
    r = _infer_r_from_len(d)

    # reshape to Pauli index tensor: m1..mr each in {0,1,2,3}
    X = vecR.reshape((4,) * r)

    # apply (T^\dagger) along each site to go from Pauli-index to pair-index (rowbit,colbit)
    Tdag = _T.conj().T
    for j in range(r):
        X = np.moveaxis(X, j, 0)                 # (4, ...)
        X = np.tensordot(Tdag, X, axes=(1, 0))   # (4, ...)
        X = np.moveaxis(X, 0, j)

    # now X is in pair-index per site (a=0..3 corresponds to (rowbit, colbit) ordering as in your perm)
    # split each a into (rowbit, colbit): shape (2,2,2,2,...)
    X = X.reshape((2, 2) * r)

    # reorder axes to matrix order: (rowbits..., colbits...)
    rows = list(range(0, 2*r, 2))
    cols = list(range(1, 2*r, 2))
    X = np.transpose(X, axes=rows + cols).reshape((1 << r), (1 << r))

    # We want W^T for fast contraction vdot(M, W^T M)
    return X.T.copy()

def C_general_fast(vecR: np.ndarray, psi: np.ndarray, r: int | None = None, k: float = 0.0, *,
                   Wt: np.ndarray | None = None) -> DTYPE:
    """
    Compute C = vdot( state_vec_Pauli(psi,r,k), vecR ) WITHOUT building the 4^r vector.
    - psi: full state vector of length 2^L (can be shaped [2]*L too)
    - vecR: length 4^r
    - k: momentum
    Optionally pass precomputed Wt = W^T (2^r x 2^r) from precompute_Wt_from_vecR(vecR).
    """
    vecR = np.asarray(vecR, dtype=DTYPE)
    d = vecR.size
    r_from = _infer_r_from_len(d)
    if r is None:
        r = r_from
    if r != r_from:
        raise ValueError("r does not match vecR length.")

    if Wt is None:
        Wt = precompute_Wt_from_vecR(vecR)   # shape (2^r, 2^r)

    psi = np.asarray(psi, dtype=DTYPE)
    L = int(np.log2(psi.size))
    if (1 << L) != psi.size:
        raise ValueError("psi.size must be a power of 2.")
    psi_t = psi.reshape((2,) * L)

    acc = np.array(0.0 + 0.0j, dtype=DTYPE)

    # For each translation i, bring the r-block to the front:
    # psi_i = transpose(psi_t, roll(axes, -i)) like your code
    axes = np.arange(L)
    dim_env = 1 << (L - r)
    dim_blk = 1 << r

    if k == 0.0:
        for i in range(L):
            psi_i = np.transpose(psi_t, np.roll(axes, -i))
            M = psi_i.reshape(dim_blk, dim_env)        # view or cheap copy
            # scalar_i = vdot(M, W^T M)  (no rho matrix allocation)
            B = Wt @ M                                  # (2^r, 2^{L-r})
            acc += np.vdot(M, B)
    else:
        for i in range(L):
            psi_i = np.transpose(psi_t, np.roll(axes, -i))
            M = psi_i.reshape(dim_blk, dim_env)
            B = Wt @ M
            acc += np.exp(-1j * k * i).astype(DTYPE) * np.vdot(M, B)

    return acc

def GHZ(L,theta):
    out = np.zeros(2**L, dtype=DTYPE)
    c = np.cos(theta*np.pi/2); s = np.sin(theta*np.pi/2)
    out[0]=c; out[-1]=s
    return out.reshape([2]*L)

if __name__=="__main__":
    r=12
    L=24
    theta_=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    vR = np.load(f"../data/vR_k=0_r={r}_c64.npy", mmap_mode="r")
    Wt  = precompute_Wt_from_vecR(vR)   

    for i,theta in enumerate(theta_):
        psi=GHZ(L,theta)
        dat=abs(C_general_fast(vR, psi, k=0.0, Wt=Wt))
        fn=f"../data/c10_GHZ_theta={theta}PI_r={r}_L={L}.npy"
        print(f"theta={theta}PI |c10|={dat:10.6f}",flush=True)
        np.save(fn,np.array([dat]))
        print(f"saved in {fn}",flush=True)
    
