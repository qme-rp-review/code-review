import numpy as np

def varrho_HS_norm(vL, ell):
    d = vL.size
    r = (d.bit_length() - 1) // 2   # d=4^r=2^(2r) を仮定
    VL = vL.reshape((4,) * r)

    acc = 0.0 + 0.0j
    for m in range(1, ell + 1):
        v = VL[(slice(None),) * m + (3,) * (r - m)].ravel()
        acc += np.vdot(v, v)
    return float(np.sqrt((acc.real / (2**ell)).clip(min=0.0)))

if __name__=="__main__":
    r=12
    ell=4

    vL=np.load(f"../data/vL_k=0_r={r}_c64.npy")
    dat=varrho_HS_norm(vL,ell)
    fn=f"../data/varrho_HS_norm_r={r}_ell={ell}.npy"
    np.save(fn,np.array([dat]))
    print(f"saved in {fn}")
        
