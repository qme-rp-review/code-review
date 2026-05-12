import numpy as np
import functools, builtins
print = functools.partial(builtins.print, flush=True)

DTYPE = np.complex64
_v0 = np.array([0, 0,  1, 1], dtype=DTYPE)
_v1 = np.array([0, 0, -1, 1], dtype=DTYPE)

def legendre_symbol(a, p):
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1

def legendre_bits(L, p=37):
    seq = [int(legendre_symbol(n, p) == -1) for n in range(1, p)]
    return np.array(seq[:L], dtype=np.uint8)

def inner_u_vR_bits(vecR: np.ndarray, bits: np.ndarray, r: int, k: float = 0.0,
                    buf1: np.ndarray | None = None, buf2: np.ndarray | None = None) -> DTYPE:
    """
    Compute <u|vecR> where u = state_vec_Pauli_from_bits(bits,r,k) (your definition),
    but without building u and without pvec.

    vecR: length 4**r, complex64
    bits: length L, 0/1 uint8
    """
    bits = np.asarray(bits, dtype=np.uint8)
    L = bits.size
    N = 4**r
    if vecR.size != N:
        raise ValueError("vecR size must be 4**r")

    # local 4-vectors per site
    p_site = np.empty((L, 4), dtype=DTYPE)
    p_site[bits == 0] = _v0
    p_site[bits == 1] = _v1

    # buffers for reductions (size 4**(r-1))
    if r == 0:
        return np.conjugate(np.array([1], dtype=DTYPE)[0]) * vecR[0]
    Mmax = 4**(r - 1)
    if buf1 is None or buf1.size < Mmax:
        buf1 = np.empty(Mmax, dtype=DTYPE)
    if buf2 is None or buf2.size < Mmax:
        buf2 = np.empty(Mmax, dtype=DTYPE)

    def contract_window(start: int) -> DTYPE:
        # returns sum_{a1..ar} conj(prod_j p_site[start+j][aj]) * vecR[a1..ar]
        X = vecR
        use1 = True
        for j in range(r - 1, -1, -1):
            v = np.conjugate(p_site[(start + j) % L])  # length 4
            X2 = X.reshape(-1, 4)
            out = (buf1 if use1 else buf2)[:X2.shape[0]]
            np.einsum("na,a->n", X2, v, out=out, optimize=True)
            X = out
            use1 = not use1
        return X[0]

    acc = np.array(0.0 + 0.0j, dtype=DTYPE)
    if k == 0.0:
        for i in range(L):
            acc += contract_window(i)

    else:
        for i in range(L):
            acc += np.exp(-1j * k * i).astype(DTYPE) * contract_window(i)

    return acc

if __name__ == "__main__":
    r = 12
    L = 24

    vR = np.load(f"../data/vR_k=0_r={r}_c64.npy", mmap_mode="r")

    p_list = [37,79,101]

    buf1 = np.empty(4**(r-1), dtype=DTYPE)
    buf2 = np.empty(4**(r-1), dtype=DTYPE)

    for p in p_list:
        bits = legendre_bits(L, p)
        val = inner_u_vR_bits(vR, bits, r, k=0.0, buf1=buf1, buf2=buf2)
        dat = float(np.abs(val))
        fn = f"../data/c10_legendre_p={p}_r={r}_L={L}.npy"
        print(f"p={p}  |<u|vR>|={dat:10.6f}")
        np.save(fn, np.array([dat], dtype=np.float32))
        print(f"saved in {fn}")