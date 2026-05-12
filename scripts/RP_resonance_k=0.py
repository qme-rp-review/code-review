import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, eigs

DTYPE = np.complex64

# ---- progress (matvec calls) ----
class MatvecProgress:
    def __init__(self, tag, every=50):
        self.tag = tag
        self.every = every
        self.calls = 0
        self.t0 = time.perf_counter()
        self.last = self.t0

    def wrap(self, f):
        def g(x):
            y = f(x)
            self.calls += 1
            if self.calls % self.every == 0:
                now = time.perf_counter()
                elapsed = now - self.t0
                dt = now - self.last
                self.last = now
                avg = elapsed / self.calls
                rate = self.every / dt if dt > 0 else float("inf")
                print(f"[{self.tag}] matvec={self.calls:7d}  elapsed={elapsed:8.1f}s  avg={avg:7.4f}s  recent={rate:6.2f}/s", flush=True)
            return y
        return g

# ---- small in-place 2x2 rotation for array slices ----
def rot2_inplace(A, B, c, s):
    # A,B are array views with same shape
    tmp = A.copy()
    A[...] = c * tmp - s * B
    B[...] = s * tmp + c * B

# ---- E operator matvecs (k=0) ----
def make_E_and_EH_k0(r, tau, hx, hz):
    N = 4**r
    dims = (4,) * (r + 2)   # padded tensor: r middle + 2 boundaries

    # trig in float32
    cx = np.float32(np.cos(2 * tau * hx)); sx = np.float32(np.sin(2 * tau * hx))
    cz = np.float32(np.cos(2 * tau * hz)); sz = np.float32(np.sin(2 * tau * hz))
    czz = np.float32(np.cos(2 * tau));     szz = np.float32(np.sin(2 * tau))

    # -tau
    cxm, sxm = cx, np.float32(-sx)
    czm, szm = cz, np.float32(-sz)
    czzm, szzm = czz, np.float32(-szz)

    # mask_Sr_indices(r) は axis0!=3 と等価 -> flattenで最後1/4を0
    tail = 3 * (4**(r-1)) if r >= 1 else 0

    buf = np.zeros(4**(r+2), dtype=DTYPE)   # 16*N
    T = buf.reshape(dims)

    # Ezz の回転ペア（16 basis -> (a0,a1))
    pairs_zz = [((8//4, 8%4), (13//4, 13%4)),
                ((2//4, 2%4), (7//4,  7%4)),
                ((12//4,12%4),(9//4,  9%4)),
                ((3//4, 3%4), (6//4,  6%4))]

    def apply_two_site_zz(T, i, c, s):
        # bring (i,i+1) to front as (4,4,rest...)
        Y = np.moveaxis(T, (i, i+1), (0, 1))
        for (a0,a1),(b0,b1) in pairs_zz:
            rot2_inplace(Y[a0, a1, ...], Y[b0, b1, ...], c, s)

    def apply_single_site_Uxz(T, i, cz, sz, cx, sx):
        # bring site i to front: (4, rest...)
        Y = np.moveaxis(T, i, 0)
        # Ez: rows 0<->1
        rot2_inplace(Y[0, ...], Y[1, ...], cz, sz)
        # Ex: rows 1<->2
        rot2_inplace(Y[1, ...], Y[2, ...], cx, sx)

    def apply_single_site_UxzH(T, i, czm, szm, cxm, sxm):
        Y = np.moveaxis(T, i, 0)
        # UxzH = Ez(-) @ Ex(-) => Ex(-) first then Ez(-)
        rot2_inplace(Y[1, ...], Y[2, ...], cxm, sxm)  # Ex(-)
        rot2_inplace(Y[0, ...], Y[1, ...], czm, szm)  # Ez(-)

    def extract_k0(T):
        # w = v[3,...,3] + v[3,3,...] + v[...,3,3] (k=0)
        out = T[(3,) + (slice(None),)*r + (3,)].reshape(N).copy()
        out += T[(3,3) + (slice(None),)*r].reshape(N)
        out += T[(slice(None),)*r + (3,3)].reshape(N)
        if tail:
            out[tail:] = 0
        return out

    def apply_E(vec):
        buf.fill(0)

        x = vec.astype(DTYPE, copy=False)
        if tail:
            tmp = x.copy()
            tmp[tail:] = 0
            mid = tmp.reshape((4,)*r)
        else:
            mid = x.reshape((4,)*r)

        # embed: kron(|3>, mid, |3>) : only this slice is nonzero
        T[(3,) + (slice(None),)*r + (3,)] = mid

        # ZZ bonds i=0..r
        for i in range(r+1):
            apply_two_site_zz(T, i, czz, szz)

        # Uxz on all sites 0..r+1
        for i in range(r+2):
            apply_single_site_Uxz(T, i, cz, sz, cx, sx)

        return extract_k0(T)

    def apply_EH(vec):
        buf.fill(0)

        mid = vec.astype(DTYPE, copy=False).reshape((4,)*r)
        T[(3,) + (slice(None),)*r + (3,)] = mid

        # UxzH on all sites
        for i in range(r+2):
            apply_single_site_UxzH(T, i, czm, szm, cxm, sxm)

        # ZZ(-tau)
        for i in range(r+1):
            apply_two_site_zz(T, i, czzm, szzm)

        # original code does w[3,...]=0 before return -> same as tail zeroing
        return extract_k0(T)

    return apply_E, apply_EH, tail

def solve_k0_largest(r, tau=0.65, hx=0.9, hz=0.8,
                     tol=1e-7, maxiter=2000, ncv=60,
                     seed=0, progress_every=50):
    N = 4**r
    apply_E, apply_EH, tail = make_E_and_EH_k0(r, tau, hx, hz)

    progR = MatvecProgress("R", every=progress_every)
    progL = MatvecProgress("L", every=progress_every)

    Lop  = LinearOperator((N, N), matvec=progR.wrap(apply_E),  dtype=DTYPE)
    LopH = LinearOperator((N, N), matvec=progL.wrap(apply_EH), dtype=DTYPE)

    rng = np.random.default_rng(seed)
    v0 = (rng.standard_normal(N) + 1j*rng.standard_normal(N)).astype(DTYPE)
    if tail:
        v0[tail:] = 0

    print(f"--- RIGHT eigs (r={r}, N={N}) ---", flush=True)
    t0 = time.perf_counter()
    valsR, vecsR = eigs(Lop, k=1, which="LM", v0=v0, tol=tol, maxiter=maxiter,
                        ncv=min(ncv, N-1) if N > 2 else None)
    tR = time.perf_counter() - t0
    lam = valsR[0].astype(DTYPE)
    vR  = vecsR[:,0].astype(DTYPE, copy=False)
    print(f"[R] done. wall={tR:.1f}s calls={progR.calls} |lam|={abs(lam):.8f}", flush=True)

    print(f"--- LEFT eigs (r={r}, N={N}) ---", flush=True)
    t0 = time.perf_counter()
    valsL, vecsL = eigs(LopH, k=1, which="LM", v0=v0, tol=tol, maxiter=maxiter,
                        ncv=min(ncv, N-1) if N > 2 else None)
    tL = time.perf_counter() - t0
    vL  = vecsL[:,0].astype(DTYPE, copy=False)
    print(f"[L] done. wall={tL:.1f}s calls={progL.calls}", flush=True)

    # biorthonormalize <vL|vR>=1
    c = np.sqrt(np.vdot(vL, vR))
    vL = vL / np.conjugate(c)
    vR = vR / c

    return lam, vL, vR

if __name__ == "__main__":
    r = 12
    lam, vL, vR = solve_k0_largest(r, tol=1e-7, ncv=60, progress_every=20)
    print("lam =", lam, " |lam| =", abs(lam), flush=True)
    np.save(f"../data/lam_k=0_r={r}_c64.npy", np.array([lam], dtype=DTYPE))
    np.save(f"../data/vL_k=0_r={r}_c64.npy", vL)
    np.save(f"../data/vR_k=0_r={r}_c64.npy", vR)