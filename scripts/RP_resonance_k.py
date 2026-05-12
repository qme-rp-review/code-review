import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import matplotlib.pyplot as plt
import functools, builtins
print = functools.partial(builtins.print, flush=True)

DTYPE = np.complex64
para = {"r":12,"tau":0.65,"hx":0.9,"hz":0.8,"k":0.0}

def Ex(tau,hx):
    c = np.cos(2*tau*hx); s = np.sin(2*tau*hx)
    M = np.eye(4, dtype=DTYPE)
    M[1,1]=c; M[2,2]=c; M[1,2]=-s; M[2,1]=s
    return M

def Ez(tau,hz):
    c = np.cos(2*tau*hz); s = np.sin(2*tau*hz)
    M = np.eye(4, dtype=DTYPE)
    M[0,0]=c; M[1,1]=c; M[0,1]=-s; M[1,0]=s
    return M

def Ezz(tau):
    c = np.cos(2*tau); s = np.sin(2*tau)
    M = np.eye(16, dtype=DTYPE)
    for a,b in [(8,13),(2,7),(12,9),(3,6)]:
        M[a,a]=c; M[b,b]=c; M[a,b]=-s; M[b,a]=s
    return M

def mask_Sr_indices(r):
    N = 4**r
    return np.fromiter(((i // (4**(r-1))) % 4 != 3 for i in range(N)), bool, N)

def GHZ(L,theta):
    out = np.zeros(2**L, dtype=DTYPE)
    c = np.cos(theta*np.pi/2); s = np.sin(theta*np.pi/2)
    out[0]=c; out[-1]=s
    return out.reshape([2]*L)

def deBruijn_bits(r_db):
    k = 2
    a = [0]*(k*r_db)
    seq = []
    def db(t,p):
        if t>r_db:
            if r_db%p==0: seq.extend(a[1:p+1])
        else:
            a[t]=a[t-p]; db(t+1,p)
            for j in range(a[t-p]+1,k):
                a[t]=j; db(t+1,t)
    db(1,1)
    return np.array(seq, dtype=np.uint8)  # length L=2**r_db

def random_bits(L):
    return np.random.randint(0,2,size=L,dtype=np.uint8)
def Neel_bits(L):
    return np.tile(np.array([0,1],dtype=np.uint8),L//2+1)[:L]

_T = np.array([[0,1,1,0],[0,1j,-1j,0],[1,0,0,-1],[1,0,0,1]], dtype=DTYPE)

def state_vec_Pauli_from_bits(bits, r, k):
    L = bits.size
    v0 = np.array([0,0, 1,1], dtype=DTYPE)
    v1 = np.array([0,0,-1,1], dtype=DTYPE)
    local = (v0, v1)
    pvec = np.empty((1<<r, 4**r), dtype=DTYPE)
    for pat in range(1<<r):
        v = np.array([1.0], dtype=DTYPE)
        for j in range(r):
            b = (pat >> (r-1-j)) & 1
            v = np.kron(v, local[b])
        pvec[pat] = v
    res = np.zeros(4**r, dtype=DTYPE)
    pow2 = 1 << np.arange(r-1, -1, -1, dtype=np.int64)
    for i in range(L):
        idxs = (i + np.arange(r)) % L
        pat = int(bits[idxs] @ pow2)
        res += np.exp(1j*k*i) * pvec[pat]
    return res / L

def state_vec_Pauli(psi,r,k):
    L = int(np.log2(psi.size))
    psi_t = psi.reshape([2]*L)
    axes_ = np.arange(L)
    perm = [q for ij in zip(range(r),range(r,2*r)) for q in ij]
    res = np.zeros(4**r, dtype=DTYPE)
    for i in range(L):
        psi_i = np.transpose(psi_t, np.roll(axes_, -i))
        M = psi_i.reshape(2**r, 2**(L-r))
        rho = (M @ M.conj().T).reshape([2]*(2*r))
        rho = np.transpose(rho, axes=perm).reshape([4]*r)
        for j in range(r):
            rho = np.moveaxis(rho, j, 0)
            rho = np.einsum("ma,a...->m...", _T, rho, optimize=True)
            rho = np.moveaxis(rho, 0, j)
        res += np.exp(1j*k*i) * rho.reshape(-1) / L
    return res

def XYZIII(vec,l):
    L = int(np.log2(np.sqrt(vec.shape[0])))
    v = vec.reshape([4]*L)
    return v[(slice(None),)*l + (3,)*(L-l)].reshape(-1)

def C(vecL,vecR,psi_0,ell,k, *, bits=None):
    d = vecL.shape[0]
    r = int(np.log(d)/2/np.log(2))
    if bits is not None:
        psi0 = state_vec_Pauli_from_bits(bits, r, k)[:d]
    else:
        psi0 = state_vec_Pauli(psi_0, r, k)[:d]
    psi0 *= mask_Sr_indices(r)[:d]
    acc = 0.0+0.0j
    VL = vecL.reshape([4]*r)
    for m in range(1,ell+1):
        v = VL[(slice(None),)*m + (3,)*(r-m)].reshape(-1)
        acc += np.vdot(v, v)
    denom = (2**ell) * (abs(np.vdot(vecL,vecR))**2)
    numer = acc * (abs(np.vdot(psi0,vecR))**2)
    return float(np.sqrt(numer/denom).real)

def _make_apply_E_family(r, tau, hx, hz):
    mask = mask_Sr_indices(r).astype(DTYPE, copy=False)
    dims = (4,)*(r+2)
    I_vec = np.zeros(4, dtype=DTYPE); I_vec[3]=1
    Uxz  = Ex(tau,hx) @ Ez(tau,hz)
    Uzz  = Ezz(tau)
    UxzH = Ez(-tau,hz) @ Ex(-tau,hx)
    UzzH = Ezz(-tau)
    def apply_E(vec):
        k = para["k"]
        v = np.kron(np.kron(I_vec, vec*mask), I_vec).reshape(dims)
        for i in range(r+1):
            v = np.moveaxis(v,(i,i+1),(0,1))
            v = (Uzz @ v.reshape(16,-1)).reshape(dims)
            v = np.moveaxis(v,(0,1),(i,i+1))
        for i in range(r+2):
            v = np.moveaxis(v,i,0)
            v = (Uxz @ v.reshape(4,-1)).reshape(dims)
            v = np.moveaxis(v,0,i)
        w = v[3,...,3] + np.exp(1j*k)*v[3,3,...] + np.exp(-1j*k)*v[...,3,3]
        return (w.reshape(-1) * mask)
    def apply_EH(vec):
        k = para["k"]
        v = np.kron(np.kron(I_vec, vec), I_vec).reshape(dims)
        for i in range(r+2):
            v = np.moveaxis(v,i,0)
            v = (UxzH @ v.reshape(4,-1)).reshape(dims)
            v = np.moveaxis(v,0,i)
        for i in range(r+1):
            v = np.moveaxis(v,(i,i+1),(0,1))
            v = (UzzH @ v.reshape(16,-1)).reshape(dims)
            v = np.moveaxis(v,(0,1),(i,i+1))
        w = v[3,...,3] + np.exp(1j*k)*v[3,3,...] + np.exp(-1j*k)*v[...,3,3]
        w = w.copy()
        w[3, ...] = 0
        return w.reshape(-1)
    return apply_E, apply_EH


def main_lam_k():
    L = 11
    r = para["r"]
    N = 4**r
    ks = np.linspace(0,0.01,L,endpoint=True)
    dat_lam =np.zeros((L,2),dtype=DTYPE);dat_lam[:,0]=ks
    
    apply_E, apply_EH = _make_apply_E_family(r, para["tau"], para["hx"], para["hz"])
    Lop  = LinearOperator((N,N), matvec=apply_E,  dtype=DTYPE)

    v0_R = None
    for i, k in enumerate(ks):
        para["k"] = k
        valsR, vecsR = eigs(Lop , k=1, which="LM", v0=v0_R, tol=1e-8, maxiter=2000)
        dat_lam[i,1] = valsR[0]
        print("----------------")
        print("k={}: |lam|={}".format(k,abs(valsR[0])))
    np.save("../data/lam_k=0to0.01_r={}".format(r),dat_lam)

if __name__=="__main__":
    main_lam_k()
    
