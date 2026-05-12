import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')
sm_data_generate=False
data_generate=False

scripts = [
    "fidelity.py",
    "RP_resonance_k=0.py",
    "RP_resonance_k.py",
    "c10_GHZ.py",
    "c10_legendre.py",
    "varrho_HS_norm.py"]

if sm_data_generate:
    scripts.append("fidelity_deBruijn.py")

if data_generate:
    for script in scripts:
        print(f"Running {script} ...")
        subprocess.run([sys.executable, script], check=True)


import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import functools, builtins
from tqdm import tqdm
from scipy.special import gamma
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
print = functools.partial(builtins.print, flush=True)
from scipy.interpolate import make_interp_spline



color_=["#3B4CC0", "#1B9E77","#D95F02","#8E3B46","#444444"]
marker_=["o","^","s","v","d","<",">"]

#fig.3
r=12
L=24
ell=4
theta_ = np.array([0.2,0.5,0.7,0.9])
plt.figure(figsize=(16,6))
#a#
plt.subplot(2,2,1)
theta__ = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
dat = np.sqrt(2-(abs(np.cos(theta__*np.pi/2))+abs(np.sin(theta__*np.pi/2)))/2)
xs = np.linspace(0, 1, 1000,endpoint=True)
ys = make_interp_spline(theta__, dat, k=3)(xs) 
dat = np.sqrt(2-(abs(np.cos(theta_*np.pi/2))+abs(np.sin(theta_*np.pi/2)))/2)
plt.plot(xs,ys,zorder=1,linewidth=2,color="k")
for i,theta in enumerate(theta_):
    plt.scatter(theta_[i],dat[i],s=100,color=color_[i])
plt.xlim(0,1)
plt.xticks([0,0.2,0.4,0.6,0.8,1],[],fontsize=13)
plt.yticks(fontsize=15)
#plt.xlabel(r"$\theta/\pi$",fontsize=20)
plt.ylabel(r"$D(0)$",fontsize=20)
plt.text(-0.15,1.22,r"$\mathbf{a}$",fontsize=25)

#b#
plt.subplot(2,2,3)
theta__ = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
dat=np.array([np.load(f"../data/c10_GHZ_theta={theta}PI_r={r}_L={L}.npy")[0]/L for theta in theta__])
xs = np.linspace(0, 1, 1000,endpoint=True)
ys = make_interp_spline(theta__, dat, k=3)(xs) 
dat=np.array([np.load(f"../data/c10_GHZ_theta={theta}PI_r={r}_L={L}.npy")[0]/L for theta in theta_])
plt.plot(xs,ys,zorder=1,linewidth=2,color="k")
for i,theta in enumerate(theta_):
    plt.scatter(theta_[i],abs(dat[i]),s=100,color=color_[i])
plt.xlim(0,1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$\theta/\pi$",fontsize=20)
plt.ylabel(r"$|c_{1,0}|$",fontsize=20)
plt.text(-0.15,0.325,r"$\mathbf{b}$",fontsize=25)

#c#
plt.subplot(2,2,(2,4))

dat1=[np.load(f"../data/fidelity_GHZ_L=24_theta={theta}pi_ell=4_tau=0.65_hx=0.9_hz=0.8.npy") for theta in theta_]

for i,theta in enumerate(theta_):
    plt.plot(np.sqrt(abs(2-2*dat1[i])),linewidth=3,label=fr"$\theta={theta}\pi$",color=color_[i])
plt.xlim(0,16)
plt.ylim(0,)
plt.xlabel(r"$t$",fontsize=20)
plt.ylabel(r"$D(t)$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.text(-2,1.2,r"$\mathbf{c}$",fontsize=25)
plt.legend(edgecolor="none",facecolor="none",loc="lower left",fontsize=15,bbox_to_anchor=(-0.01,-0.02))
main_plot=plt.gca()

#inset of c"
ins = inset_axes(main_plot,width="55%", height="55%",loc="upper right",borderpad=1)
plt.sca(ins)
t_max=60
t_min=1
t_=np.arange(t_min,t_max+1)
HS=np.load(f"../data/varrho_HS_norm_r={r}_ell=4.npy")[0]
lam_k=np.load(f"../data/lam_k=0to0.01_r={r}.npy")
lam=abs(lam_k[0,1])
c10_GHZ=[np.load(f"../data/c10_GHZ_theta={theta}PI_r={r}_L=24.npy")[0] for theta in theta_]
for i,theta in enumerate(theta_):
    plt.plot(np.sqrt(2-2*dat1[i]),label=fr"$\theta={theta}\pi$",zorder=1,color=color_[i],linewidth=3)
    plt.plot(t_,c10_GHZ[i]*lam**t_*HS*2**(ell/2-1)/L,linestyle="dotted",color=color_[i],linewidth=2)
plt.ylim(0.01,)
plt.xlim(0,60)
plt.yscale("log")
plt.xlabel("$t$",fontsize=20)
plt.ylabel("$D(t)$",fontsize=20)

plt.tight_layout()
plt.savefig("../figures/fig3.pdf",bbox_inches="tight")

#fig.4
plt.figure()
r=12
L=24
ell=4
theta_=[0.5]
p_=[37,79,101]
t_min=0
t_max=16
dat1=[np.load(f"../data/fidelity_GHZ_L=24_theta={theta}pi_ell=4_tau=0.65_hx=0.9_hz=0.8.npy") for theta in theta_]
dat2=[np.load(f"../data/fidelity_legendre_p={p}_L=24_ell=4_tau=0.65_hx=0.9_hz=0.8.npy") for p in p_]

for i,theta in enumerate(theta_):
    plt.plot(np.sqrt(2-2*dat1[i][t_min:t_max+1]),linestyle="dashed",color=color_[0],linewidth=2)
for i,p in enumerate(p_):
    plt.plot(np.sqrt(2-2*dat2[i][t_min:t_max+1]),label=rf"$p={p}$",color=color_[i+1],linewidth=2)
plt.xlim(t_min,t_max)
plt.ylim(0,)
plt.xlabel(r"$t$",fontsize=20)
plt.ylabel(r"$D(t)$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=13,edgecolor="none",facecolor="none",loc="lower left",bbox_to_anchor=(-0.02,-0.02))
main_plot=plt.gca()

ins = inset_axes(main_plot,width="60%", height="60%",loc="upper right",borderpad=1)
plt.sca(ins)
t_min=0
t_max=60
t_=np.arange(t_min,t_max+1)
HS=np.load(f"../data/varrho_HS_norm_r={r}_ell=4.npy")[0]
lam_k=np.load(f"../data/lam_k=0to0.01_r={r}.npy")
lam=abs(lam_k[0,1])
c10_GHZ=[np.load(f"../data/c10_GHZ_theta={theta}PI_r={r}_L=24.npy")[0] for theta in theta_]
c10_leg=[np.load(f"../data/c10_legendre_p={p}_r={r}_L={L}.npy")[0] for p in p_]
def f(x,a):
    return a*x**2
x = abs(lam_k[:,0])
y = -np.log(abs(lam_k[:,1]))
popt,_=curve_fit(f,x,y-y[0])
phi=popt[0]
for i,p in enumerate(p_):
    plt.plot(t_,np.sqrt(2-2*dat2[i][t_min:t_max+1]),label=fr"$p={p}$",color=color_[i+1],linewidth=2)
    plt.plot(t_[1:],abs(2**(ell/2-1)*lam**t_[1:]*HS*abs(c10_leg[i]))/np.sqrt(2*np.pi*t_[1:]*phi),linestyle="dotted",color=color_[i+1],linewidth=2)
plt.xlabel(r"$t$",fontsize=20)
plt.ylabel(r"$D(t)$",fontsize=20)
plt.yscale("log")
plt.xlim(t_min,t_max)

plt.savefig("../figures/fig4.pdf",bbox_inches="tight")

plt.figure()
L = 24
r = 12
theta_=[0.2,0.5,0.7,0.9]

HS=np.load(f"../data/varrho_HS_norm_r={r}_ell=4.npy")[0]
lam = abs(np.load(f"../data/lam_k=0_r={r}_c64.npy")[0])
c10_ = [np.load(f"../data/c10_GHZ_theta={theta}PI_r={r}_L=24.npy")[0] for theta in theta_]
dat1_ = [np.load(f"../data/fidelity_GHZ_L={L}_theta={theta}pi_ell=4_tau=0.65_hx=0.9_hz=0.8.npy") for theta in theta_]
dat2 = np.load("../data/fidelity_deBruijn_r=5.npy")
t_ = np.arange(len(dat2))
plt.plot(np.sqrt(2-2*dat2),label=r"$D(t)$",color=color_[0],linewidth=2)
plt.plot(np.sqrt(2-2*dat2)/lam**t_,label=r"$D(t)\times\lambda_{1,0}^{-t}$",color=color_[1],linewidth=2)
plt.plot(t_[1:],np.sqrt(2-2*dat2[1:])/lam**t_[1:]*t_[1:]**0.5,label=r"$D(t)\times \lambda_{1,0}^{-t} t^{1/2}$",color=color_[2],linewidth=2)
plt.plot(t_[1:],np.sqrt(2-2*dat2[1:])/lam**t_[1:]*t_[1:]**1,label=r"$D(t)\times \lambda_{1,0}^{-t} t$",color=color_[3],linewidth=2)
plt.yscale("log")
plt.xlim(0,35)
plt.ylim(1e-2,)
plt.xlabel(r"$t$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(facecolor="white",edgecolor="black",framealpha=1)
plt.grid()

plt.savefig("../figures/fig.sm-1.pdf",bbox_inches="tight")



