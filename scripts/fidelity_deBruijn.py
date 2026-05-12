import numpy as np
import time
import matplotlib.pyplot as plt

# =========================
# popcount for uint32 (NumPy 1.x compatible)
# =========================
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

def popcount_u32(x_u32: np.ndarray) -> np.ndarray:
    """Vectorized popcount for np.uint32 array -> np.uint8 array."""
    lo = (x_u32 & np.uint32(0xFFFF)).astype(np.uint16, copy=False)
    hi = (x_u32 >> np.uint32(16)).astype(np.uint16, copy=False)
    return (_POP16[lo] + _POP16[hi]).astype(np.uint8, copy=False)

# =========================
# de Bruijn bits (binary) of order r: length L=2^r
# =========================
def deBruijn_bits(r: int) -> np.ndarray:
    k = 2
    a = [0] * (k * r)
    seq = []
    def db(t, p):
        if t > r:
            if r % p == 0:
                seq.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return np.array(seq, dtype=np.uint8)  # length 2^r

def bits_to_index_msb(bits: np.ndarray) -> np.uint64:
    # MSB-first: idx = sum bits[i] << (L-1-i)
    L = bits.size
    idx = np.uint64(0)
    for i, b in enumerate(bits.tolist()):
        if b:
            idx |= (np.uint64(1) << np.uint64(L - 1 - i))
    return idx

def init_basis_state(L: int, idx: np.uint64, dtype=np.complex64) -> np.ndarray:
    N = np.uint64(1) << np.uint64(L)
    psi = np.zeros(int(N), dtype=dtype)
    psi[int(idx)] = dtype(1.0 + 0.0j)
    return psi  # flat, length 2^L

# =========================
# Gates
# =========================
def rot_xz(tau, hx, hz, dtype=np.complex64):
    a = tau * hx
    b = tau * hz
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)

    uz00 = cb + 1j * sb
    uz11 = cb - 1j * sb
    ux00 = ca
    ux01 = 1j * sa
    ux10 = 1j * sa
    ux11 = ca

    U = np.empty((2, 2), dtype=np.complex64)
    U[0,0] = ux00 * uz00
    U[0,1] = ux01 * uz11
    U[1,0] = ux10 * uz00
    U[1,1] = ux11 * uz11
    return U.astype(dtype, copy=False)

# =========================
# Fast ZZ on ALL bonds in ONE pass:
# phase(x) = exp(i*tau * sum_i z_i z_{i+1}), z=+1 for bit=0, z=-1 for bit=1, periodic
# sum = L - 2 * (# of adjacent differences in circular bitstring)
# =========================
def apply_ZZ_all_inplace(psi: np.ndarray, tau: float, L: int, chunk: int = 1 << 24):
    """
    Works for L<=32. Uses uint32 index trick and popcount LUT (NumPy 1.x OK).
    """
    assert L <= 32
    N = psi.size

    # mask for bits 0..L-2 of (idx ^ (idx>>1))
    if L == 1:
        return
    mask = np.uint32((1 << (L - 1)) - 1)  # fits for L<=32

    for base in range(0, N, chunk):
        end = min(base + chunk, N)
        idx = np.arange(base, end, dtype=np.uint32)

        x = idx ^ (idx >> np.uint32(1))
        diff = popcount_u32(x & mask).astype(np.int16, copy=False)

        # periodic last pair (L-1,0):
        last = (((idx >> np.uint32(L - 1)) ^ (idx & np.uint32(1))) & np.uint32(1)).astype(np.int16, copy=False)
        diff += last

        s = (np.int16(L) - (diff << 1)).astype(np.float32, copy=False)
        ang = (tau * s).astype(np.float32, copy=False)
        ph = (np.cos(ang) + 1j * np.sin(ang)).astype(np.complex64, copy=False)

        psi[base:end] *= ph

# =========================
# Fast 1-site gate in-place using reshape views, chunked temporaries
# site i corresponds to bit i (LSB convention)
# =========================
def apply_1site_inplace(psi: np.ndarray, i: int, U: np.ndarray, max_tmp_bytes: int = 256 << 20):
    N = psi.size
    stride = 1 << i
    period = stride << 1
    M = N // period

    block = psi.reshape(M, period)   # view
    v0 = block[:, :stride]           # view
    v1 = block[:, stride:]           # view

    u00, u01 = U[0, 0], U[0, 1]
    u10, u11 = U[1, 0], U[1, 1]

    item = psi.itemsize
    rows_per_chunk = max(1, max_tmp_bytes // (2 * stride * item))

    for s in range(0, M, rows_per_chunk):
        t = min(s + rows_per_chunk, M)
        a = v0[s:t]
        b = v1[s:t]
        y0 = u00 * a + u01 * b
        y1 = u10 * a + u11 * b
        v0[s:t] = y0
        v1[s:t] = y1

def time_step(psi: np.ndarray, tau: float, U: np.ndarray, L: int,
              zz_chunk: int = 1 << 24, max_tmp_bytes: int = 256 << 20):
    apply_ZZ_all_inplace(psi, tau=tau, L=L, chunk=zz_chunk)
    for i in range(L):
        apply_1site_inplace(psi, i=i, U=U, max_tmp_bytes=max_tmp_bytes)
    return psi

# =========================
# Fidelity quantity (ell fixed small, default 4)
# =========================
def fidelity_blocked(psi: np.ndarray, L: int, ell: int = 4, block_cols: int = 1 << 18):
    ell = min(ell, L - ell)
    k = 1 << ell
    A = psi.reshape(k, -1)  # view
    cols = A.shape[1]

    G = np.zeros((k, k), dtype=np.complex64)
    for s in range(0, cols, block_cols):
        t = min(s + block_cols, cols)
        B = A[:, s:t]
        G += B @ B.conj().T

    # eigvalsh is cheap (16x16). Use complex128 for numerical safety.
    eig = np.linalg.eigvalsh(G.astype(np.complex128))
    eig = np.clip(eig, 0.0, None)
    return float(np.sum(np.sqrt(eig)) / np.sqrt(k))

# =========================
# Main: deBruijn order r, length L=2^r (so r=5 -> L=32)
# =========================
def main_deBruijn(r: int,
                 T: int = 51,
                 ell: int = 4,
                 tau: float = 0.65,
                 hx: float = 0.9,
                 hz: float = 0.8,
                 zz_chunk: int = 1 << 24,
                 max_tmp_bytes: int = 256 << 20,
                 block_cols: int = 1 << 18):
    L = 1 << r
    bits = deBruijn_bits(r)

    idx = bits_to_index_msb(bits)

    print(f"r={r}  L={L}, N=2^L={1<<L:,} amplitudes")
    print(f"psi complex64 ~{(1<<L)*8/1e9:.3f} GB")

    psi = init_basis_state(L, idx, dtype=np.complex64)
    U = rot_xz(tau, hx, hz, dtype=np.complex64)

    vals = np.empty(T, dtype=np.float32)
    t_total = 0.0

    for t in range(T):
        t0 = time.perf_counter()
        f = fidelity_blocked(psi, L=L, ell=ell, block_cols=block_cols)
        vals[t] = f
        psi = time_step(psi, tau=tau, U=U, L=L, zz_chunk=zz_chunk, max_tmp_bytes=max_tmp_bytes)
        dt = time.perf_counter() - t0
        t_total += dt
        print(f"t={t:3d}  sqrt(2(1-f))={np.sqrt(2*(1-f)):.6f}   total={t_total/60:.2f} min   dt={dt:.2f} s", flush=True)

    return vals

if __name__ == "__main__":
    vals = main_deBruijn(
        r=5,   
        T=61,
        ell=4,
        tau=0.65,
        hx=0.9,
        hz=0.8,
        zz_chunk=1<<24,
        max_tmp_bytes=256<<20,
        block_cols=1<<18
    )
    np.save("fidelity_deBruijn_r=5.npy", vals)