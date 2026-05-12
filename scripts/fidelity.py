import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import functools, builtins
import time
from scipy.optimize import least_squares
print = functools.partial(builtins.print, flush=True)
DTYPE=complex

def mask_Sr_indices(r):
    N = 4**r
    return np.fromiter(((i // (4**(r-1))) % 4 != 3 for i in range(N)), bool, N)
    
def local_xyz(theta,phi):
    x=np.sin(theta)*np.cos(phi)
    y=np.sin(theta)*np.sin(phi)
    z=np.cos(theta)
    return np.array([x,y,z,1])

def psi_ab(a,b,bits):
    bits = np.array(bits,dtype=np.int8)
    L=bits.size
    temp_a = np.array([np.cos(a[0]/2),np.sin(a[0]/2)*np.exp(1j*a[1])])
    temp_b = np.array([np.cos(b[0]/2),np.sin(b[0]/2)*np.exp(1j*b[1])])
    if bits[0]==0:
        ret = temp_a
    else:
        ret = temp_b
    for i in range(1,L):
        if bits[i]==0:
            ret = np.kron(ret,temp_a)
        else:
            ret = np.kron(ret,temp_b)
    return ret.reshape([2]*L)

def state_vec_Pauli_from_product(a,b,bits,r):
    bits = np.array(bits,dtype=np.int8)
    L=bits.size
    pa = local_xyz(a[0],a[1])
    pb = local_xyz(b[0],b[1])
    p_site=np.empty((L,4))
    p_site[bits==0]=pa
    p_site[bits==1]=pb
    ret = np.zeros(4**r)
    for i in range(L):
        vec = p_site[i]
        for j in range(1,r):
            vec = (vec[:,None]*p_site[(i+j)%L][None,:]).reshape(-1)
        ret+=vec
    return ret

def solve_ab(bits,r,vR):
    def func(params):
        theta_a,phi_a,theta_b,phi_b=params
        a=[theta_a,phi_a];b=[theta_b,phi_b]
        ret = np.vdot(state_vec_Pauli_from_product(a,b,bits,r),vR)
        return np.array([ret.real,ret.imag])
    x0=np.array([1.0,0.0,2.0,1.0])
    lb = np.array([0,-np.pi,0,-np.pi])
    ub = np.array([np.pi,np.pi,np.pi,np.pi])

    sol = least_squares(func,x0,bounds=(lb,ub))
    theta_a,phi_a,theta_b,phi_b=sol.x
    a=[theta_a,phi_a];b=[theta_b,phi_b]
    ret = np.vdot(state_vec_Pauli_from_product(a,b,bits,r),vR)
    return a,b,ret

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
    return res/L


def rot_xz(tau, hx, hz):
    a = tau*hx; b = tau*hz
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    Ux = ca*I + 1j*sa*X
    Uz = cb*I + 1j*sb*Z
    return (Ux @ Uz).astype(complex)

def zz_phase(tau):
    epp = np.exp(1j*tau)
    emm = np.exp(-1j*tau)
    return np.array([[epp, emm],[emm, epp]], dtype=complex)

def apply_ZZ_diag_inplace(psi, i, j, phase):
    sl = [slice(None)]*psi.ndim
    for si in (0,1):
        sl[i] = si
        for sj in (0,1):
            sl[j] = sj
            psi[tuple(sl)] *= phase[si, sj]

def apply_1site_blocked_gemm(psi: np.ndarray, i: int, U: np.ndarray, block_cols: int = 1<<22):
    L = psi.ndim
    N = 1 << L
    bit = 1 << i
    stride = bit
    period = bit << 1
    flat = psi.reshape(N)
    cols_per_period = stride
    pairs_per_block = max(1, block_cols // cols_per_period)
    block_span = pairs_per_block * period
    for base in range(0, N, block_span):
        span = min(block_span, N - base)
        num_pairs = span // period
        if num_pairs == 0:
            continue
        B = num_pairs * stride
        X0 = np.empty(B, dtype=flat.dtype)
        X1 = np.empty(B, dtype=flat.dtype)
        write = 0
        for p in range(num_pairs):
            i0 = base + p*period
            i1 = i0 + stride
            v0 = flat[i0:i0+stride]
            v1 = flat[i1:i1+stride]
            X0[write:write+stride] = v0
            X1[write:write+stride] = v1
            write += stride
        X = np.stack([X0, X1], axis=0)
        Y = U @ X
        read = 0
        for p in range(num_pairs):
            i0 = base + p*period
            i1 = i0 + stride
            flat[i0:i0+stride] = Y[0, read:read+stride]
            flat[i1:i1+stride] = Y[1, read:read+stride]
            read += stride

def time_step(psi, phase_ZZ, Uxz, block_cols=1<<22):
    L = psi.ndim
    for i in range(L):
        apply_ZZ_diag_inplace(psi, i, (i+1) % L, phase_ZZ)
    for i in range(L):
        apply_1site_blocked_gemm(psi, i, Uxz, block_cols=block_cols)
    return psi

def purity_blocked(psi, ell, L, block_cols: int = 1<<22):
    ell = min(ell, L-ell)
    A = psi.reshape(1<<ell, -1).astype(np.complex64, copy=False)
    k = A.shape[0]
    G = np.zeros((k, k), dtype=np.complex64)
    cols = A.shape[1]
    for s in range(0, cols, block_cols):
        t = min(s + block_cols, cols)
        B = A[:, s:t]
        G += B @ B.conj().T
    return float(np.vdot(G, G).real)

def fidelity_blocked(psi, ell, L, block_cols: int = 1<<22):
    ell = min(ell, L-ell)
    A = psi.reshape(1<<ell, -1).astype(np.complex128, copy=False)
    k = A.shape[0]
    G = np.zeros((k, k), dtype=np.complex128)
    cols = A.shape[1]
    for s in range(0, cols, block_cols):
        t = min(s + block_cols, cols)
        B = A[:, s:t]
        G += B @ B.conj().T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    result = np.sum(np.sqrt(eigvals))/np.sqrt(k)
    return result 
    

def deBruijn_bits(r):
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
    return seq

def deBruijn_state_zproduct(r):
    bits = deBruijn_bits(r)
    L = 1 << r
    idx = int(''.join(map(str, bits)), 2)
    psi = np.zeros(1 << L, dtype=complex)
    psi[idx] = 1.0
    psi = psi.reshape([2]*L)
    return np.ascontiguousarray(psi)

def deBruijn_state_repeated(r,n):
    bits = deBruijn_bits(r)
    L = 1 << r
    idx = int(''.join(map(str, bits)), 2)
    psi = np.zeros(1 << L, dtype=complex)
    psi[idx] = 1.0
    result = psi.copy()
    for i in range(n-1):
        result = np.kron(result,psi)
    result = result.reshape([2]*(L*n))
    return np.ascontiguousarray(result)

def Neel(L,theta):
    temp1 = np.array([1,0],dtype=complex)
    temp2 = np.array([0,1],dtype=complex)
    result1 = temp1.copy()
    result2 = temp2.copy()
    for i in range(L-1):
        if i%2==0:
            result1 = np.kron(result1,temp2)
            result2 = np.kron(result2,temp1)
        else:
            result1 = np.kron(result1,temp1)
            result2 = np.kron(result2,temp2)
    result = result1*np.cos(theta*np.pi/2)+result2*np.sin(theta*np.pi/2)
    result = result.reshape([2]*L)
    return np.ascontiguousarray(result)

def tilte(L):
    result = np.array([1,0],dtype=complex)
    for i in range(1,L):
        theta = 2*np.pi*i/L
        temp = np.array([np.cos(theta),np.sin(theta)],dtype=complex)
        result = np.kron(np.kron(result,temp))
    result = result.reshape([2]*L)
    return np.ascontiguousarray(result)

def Random(L):
    result = np.random.rand(2**L)+0.j
    result/= norm(result)
    result = result.reshape([2]*L)
    return np.ascontiguousarray(result)

def Random_bit(L):
    i = np.random.randint(0,2**L)
    state = np.zeros(2**L,dtype=complex)
    state[i]=1.
    state = state.reshape([2]*L)
    return np.ascontiguousarray(state)

def GHZ(L,theta):
    result = np.zeros(2**L,dtype=complex)
    result[0] = 1
    result*=np.cos(theta*np.pi/2)
    temp = np.zeros(2**L,dtype=complex)
    temp[-1]=1
    result+=temp*np.sin(theta*np.pi/2)
    result = result.reshape([2]*L)
    return np.ascontiguousarray(result)

    
def legendre_symbol(a, p):
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1

def legendre_bits(L,p=37):
    seq = [int(legendre_symbol(n,p)==-1) for n in range(1,p)]
    return seq[:L]

def legendre_state_zproduct(L,p):
    bits = legendre_bits(L,p)
    idx = int(''.join(map(str, bits)), 2)
    psi = np.zeros(1 << L, dtype=complex)
    psi[idx] = 1.0
    psi = psi.reshape([2]*L)
    return np.ascontiguousarray(psi)

def rudin_shapiro_pm1(L):
    r = np.empty(L, dtype=np.int8)
    for n in range(L):
        x = n
        parity = 0
        prev = 0
        while x:
            bit = x & 1
            if bit & prev:      # ...11... を見つけた
                parity ^= 1
            prev = bit
            x >>= 1
        r[n] = 1 if parity == 0 else -1
    return r

def rudin_shapiro_bits(L):
    r = rudin_shapiro_pm1(L)
    return ((1 - r) // 2).astype(np.int8)  # +1->0, -1->1

def rudin_shapiro_state(L):
    bits = rudin_shapiro_bits(L)
    idx = int(''.join(map(str, bits)), 2)
    psi = np.zeros(1 << L, dtype=complex)
    psi[idx] = 1.0
    psi = psi.reshape([2]*L)
    return np.ascontiguousarray(psi)

def thue_morse_bits(L: int, *, as_int=True):
    seq = [(n.bit_count() & 1) for n in range(L)]  # popcount mod 2

    if as_int:
        return seq
    else:
        return "".join("1" if x else "0" for x in seq)

def thue_morse_state(L):
    bits = thue_morse_bits(L)
    idx = int(''.join(map(str, bits)), 2)
    psi = np.zeros(1 << L, dtype=complex)
    psi[idx] = 1.0
    psi = psi.reshape([2]*L)
    return np.ascontiguousarray(psi)

def baum_sweet_bits(L: int, *, as_int=True):
    def is_baum_sweet_one(n: int) -> int:
        # Scan bits from LSB upward, tracking lengths of consecutive 0-blocks
        if n == 0:
            return 1

        run0 = 0
        while n > 0:
            if (n & 1) == 0:
                run0 += 1
            else:
                if (run0 & 1) == 1:  # odd-length zero block
                    return 0
                run0 = 0
            n >>= 1

        # If number ends with zeros, we have a trailing zero-block to check
        return 0 if (run0 & 1) == 1 else 1

    seq = [is_baum_sweet_one(n) for n in range(L)]
    return seq if as_int else "".join("1" if x else "0" for x in seq)
                                      

def state_from_bits(bits):
    L = len(bits)
    idx = int(''.join(map(str, bits)), 2)
    psi = np.zeros(1 << L, dtype=complex)
    psi[idx] = 1.0
    psi = psi.reshape([2]*L)
    return np.ascontiguousarray(psi)

def period_doubling_bits(L, *, as_int=True):
    s = "0"
    while len(s) < L:
        # apply substitution
        s = "".join("01" if c == "0" else "00" for c in s)
    s = s[:L]
    if as_int:
        return [0 if c == "0" else 1 for c in s]
    return s

def fibonacci_word_bits(L: int, *, as_int=True):
    s = "0"
    while len(s) < L:
        s = "".join("01" if c == "0" else "0" for c in s)
    s = s[:L]

    if as_int:
        return [0 if c == "0" else 1 for c in s]
    return s

def paperfolding(L: int, *, as_int=True):
    seq = [0] * L
    if L == 0:
        return [] if as_int else ""

    seq[0] = 0  # convention
    for n in range(1, L):
        m = n
        # remove all factors of 2: m becomes odd
        m >>= (m & -m).bit_length() - 1  # same as: while m%2==0: m//=2
        seq[n] = 1 if (m & 3) == 1 else 0  # m mod 4: 1 -> 1, 3 -> 0

    return seq if as_int else "".join("1" if x else "0" for x in seq)



########################################
########################################
########################################
########################################

def psi_ab_legendre(L,p,r):
    vR=np.load(f"../data/vecR1_k=0_r={r}.npy")
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 61
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    bits=legendre_bits(L,p)
    a,b,_= solve_ab(bits,r,vR)
    print(_)
    psi=psi_ab(a,b,bits)
    
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    t_tot=0
    for t in times:
        dt = time.perf_counter()
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        dt = time.perf_counter()-dt
        t_tot+=dt
        print(f"t={t} {np.sqrt(2*(1-vals[t])):10.6f} {t_tot/60:6.1f}[min] {dt:6.1f}[sec/it]",flush=True)
    fn = f"../data/fidelity_ab_legendre_L={L}_ell={ell}_tau={tau}_hx={hx}_hz={hz}_r={r}_p={p}.npy"
    np.save(fn, vals)
    print(f"saved in {fn}")
    


def test(L):
    r=10
    mask=mask_Sr_indices(r)
    r_d=5
    L=2**r_d
    bits=np.array(deBruijn_bits(r_d),dtype=np.uint8)
    
    vR=np.load(f"../data/vecR1_k=0_r={r}.npy")[mask]
    #a,b,_= solve_ab(bits,r,vR)
    u=state_vec_Pauli_from_bits(bits, r, 0)[mask]
    print(np.vdot(u,vR))

    r+=1
    mask=mask_Sr_indices(r)
    u=state_vec_Pauli_from_bits(bits, r, 0)[mask]
    vR=np.load(f"../data/vecR1_k=0_r={r}.npy")[mask]
    print(np.vdot(u,vR))
    
    
    

def main_deBruijn(r):
    L = 1 << r
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 51
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = deBruijn_state_zproduct(r)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    t_tot = 0
    for t in times:
        dt = time.perf_counter()
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        dt = time.perf_counter()-dt
        t_tot+=dt
        print(f"t={t:6.1f} {np.sqrt(2*(1-vals[t])):10.6f} {t_tot/60:10.1f}[min] {dt:10.1f}[sec/it]")
    fn = f"../data/fidelity_deBruijn_r={r}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    print(f"saved in {fn}")

def main_legendre(L,p,T=51):
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = state_from_bits(legendre_bits(L,p))
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    t_total=0
    for t in times:
        t0=time.perf_counter()
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        dt=time.perf_counter()-t0
        t_total+=dt
        print(f"t={t:6.1f} {np.sqrt(2*(1-vals[t])):10.6f} {t_total/60:10.1f}[min] {dt:10.1f}[sec/it]")
    fn = f"../data/fidelity_legendre_p={p}_L={L}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    print(f"saved in {fn}")
    #plt.plot(np.sqrt(2*np.abs(1-vals)))
    #plt.yscale("log")
    #plt.show()

def main_deBruijn_repeated():
    r = 3
    n=3
    L = 2**r*n
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 50
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = deBruijn_state_repeated(r,n)
    times = np.arange(T)
    vals = np.empty(T, dtype=complex)
    for t in tqdm(times, bar_format="{n_fmt}/{total_fmt} {rate_fmt}"):
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
    fn = f"../data/fidelity_deBruijn_repeated_r={r}_n={n}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(2*np.abs(1-vals)))
    plt.yscale("log")
    plt.show()

def main_Neel(theta):
    L=22
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 50
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = Neel(L,theta)#deBruijn_state_zproduct(r)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    for t in times:
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        print(t,vals[t])
    fn = f"../data/fidelity_Neel_L={L}_theta={theta}pi_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(np.abs(vals - 1/2**ell)))
    plt.yscale("log")
    plt.ylim(1e-3,)
    plt.show()

def main_tilte(theta):
    L=22
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 50
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = Neel(L,theta)#deBruijn_state_zproduct(r)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    for t in times:
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        print(t,vals[t])
    fn = f"../data/fidelity_Neel_L={L}_theta={theta}pi_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(np.abs(vals - 1/2**ell)))
    plt.yscale("log")
    plt.ylim(1e-3,)
    plt.show()
    
def main_Random(L):
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 100
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = Random(L)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    for t in times:
        vals[t] = purity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        print(t,vals[t])
    fn = f"../data/purity_Random_L={L}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(np.abs(vals - 1/2**ell)))
    plt.yscale("log")
    plt.ylim(1e-3,)
    plt.show()

    
def main_Random_bit(L):
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 50
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = Random_bit(L)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    for t in times:
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        print(t,np.sqrt(2*(1-vals[t])),flush=True)
    fn = f"../data/fidelity_Random_bit_L={L}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(2*np.abs(1-vals)))
    plt.yscale("log")
    plt.ylim(1e-3,)
    plt.show()

def main_rudin_shapiro(L):
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 50
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = rudin_shapiro_state(L)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    for t in times:
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        print(t,np.sqrt(2*(1-vals[t])),flush=True)
    fn = f"../data/fidelity_rudin_shapiro_L={L}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(2*np.abs(1-vals)))
    plt.yscale("log")
    plt.ylim(1e-3,)
    plt.show()

def main_thue_morse(L):
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    T = 50
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = thue_morse_state(L)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    for t in times:
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        print(t,np.sqrt(2*(1-vals[t])),flush=True)
    fn = f"../data/fidelity_thue_morse_L={L}_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    plt.plot(np.sqrt(2*np.abs(1-vals)))
    plt.yscale("log")
    plt.ylim(1e-3,)
    plt.show()

def main_GHZ(L,theta,T=101):
    ell = 4
    tau = 0.65
    hx = 0.9
    hz = 0.8
    Uxz = rot_xz(tau, hx, hz)
    phase_ZZ = zz_phase(tau)
    psi = GHZ(L,theta)#deBruijn_state_zproduct(r)
    times = np.arange(T)
    vals = np.empty(T, dtype=np.float64)
    t_tot=0
    for t in times:
        dt = time.perf_counter()
        vals[t] = fidelity_blocked(psi, ell, L, block_cols=1<<22)
        psi = time_step(psi, phase_ZZ, Uxz, block_cols=1<<22)
        dt = time.perf_counter()-dt
        t_tot+=dt
        print(f"t={t:4.1f} {np.sqrt(2*(1-vals[t])):10.6f} {t_tot:10.1f}[sec] {dt:10.1f}[sec/it]")
    fn = f"../data/fidelity_GHZ_L={L}_theta={theta}pi_ell={ell}_tau={tau}_hx={hx}_hz={hz}.npy"
    np.save(fn, vals)
    print(f"saved in {fn}")
    

if __name__ == "__main__":
    theta_=[0.2,0.5,0.7,0.9]
    for theta in theta_:
        main_GHZ(L=24,theta=theta,T=62)
    p_=[37,79,101]
    for p in p_:
        main_legendre(L=24,p=p,T=62)
    
