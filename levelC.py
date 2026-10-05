"""
Уровень C для модели Каца.
 (1) Точные конечно-N системы для моментов степени 4 и 6 (sympy): gamma_4(N), gamma_6(N).
 (2) Монте-Карло процесса Каца: скорость релаксации m4 при N=8,16,32,64.
 (3) Уравнение Ляпунова для ковариации в базисе Эрмита вдоль решения Крука--Ву.
lambda = 1.
"""
import numpy as np, sympy as sp, json, time
from numpy.polynomial.hermite_e import hermeval, hermegauss
from math import factorial

# ================= (1) точные конечно-N системы =================
th = sp.symbols('theta'); N = sp.symbols('N', positive=True)
c, s = sp.cos(th), sp.sin(th)
def avg_th(expr):
    return sp.simplify(sp.integrate(sp.expand(expr), (th, 0, 2*sp.pi))/(2*sp.pi))
v1, v2, v3, vj = sp.symbols('v1 v2 v3 vj')
# моменты как символы: m[a,b,c] = E[v1^a v2^b v3^c] (симметричны по перестановкам)
def msym(*exps):
    e = tuple(sorted([x for x in exps if x > 0], reverse=True))
    if any(x % 2 for x in e): return sp.Integer(0)
    if not e: return sp.Integer(1)
    if e == (2,): return sp.Integer(1)
    return sp.Symbol('m' + ''.join(str(x) for x in e))
def E_poly(poly, vars_):
    """E[poly] по обменной мере: заменить мономы на m-символы"""
    P = sp.Poly(sp.expand(poly), *vars_); tot = 0
    for mon, coef in P.terms(): tot += coef*msym(*mon)
    return sp.expand(tot)

def rate_generator(phi, particles, vars_):
    """d/dt E[phi] для процесса Каца: пары (i,j) с интенсивностью lam/N.
       particles: список переменных-частиц в phi; остальные частицы -> vj."""
    res = 0
    P = list(particles)
    # пары внутри phi
    for a in range(len(P)):
        for b in range(a+1, len(P)):
            x, y = P[a], P[b]
            rot = {x: x*c - y*s, y: x*s + y*c}
            res += sp.Rational(1, 1)/N * (E_poly(avg_th(phi.subs(rot, simultaneous=True)), vars_) - E_poly(phi, vars_))
    # пары (частица в phi, внешняя)
    for x in P:
        rot = {x: x*c - vj*s}
        res += (N - len(P))/N * (E_poly(avg_th(phi.subs(rot, simultaneous=True)), vars_ + [vj]) - E_poly(phi, vars_))
    return sp.simplify(res)

vars4 = [v1, v2]
eq4 = {'m4': rate_generator(v1**4, [v1], vars4), 'm22': rate_generator(v1**2*v2**2, [v1, v2], vars4)}
print('degree 4:', eq4)
m4, m22 = sp.symbols('m4 m22')
A4 = sp.Matrix([[sp.diff(eq4['m4'], m4), sp.diff(eq4['m4'], m22)], [sp.diff(eq4['m22'], m4), sp.diff(eq4['m22'], m22)]])
ev4 = [sp.simplify(e) for e in A4.eigenvals().keys()]
print('eigenvalues deg4:', ev4)
vars6 = [v1, v2, v3]
eq6 = {'m6': rate_generator(v1**6, [v1], vars6), 'm42': rate_generator(v1**4*v2**2, [v1, v2], vars6),
       'm222': rate_generator(v1**2*v2**2*v3**2, [v1, v2, v3], vars6)}
print('degree 6:', eq6)
m6, m42, m222 = sp.symbols('m6 m42 m222')
syms6 = [m6, m42, m222]
A6 = sp.Matrix([[sp.diff(eq6[k], x) for x in syms6] for k in ('m6', 'm42', 'm222')])
b6 = sp.Matrix([eq6[k].subs({x: 0 for x in syms6}) for k in ('m6', 'm42', 'm222')])
ev6 = {sp.simplify(k): v for k, v in A6.eigenvals().items()}
print('eigenvalues deg6:', ev6)
# разложение по 1/N
ev6_ser = [sp.series(e, N, sp.oo, 3) for e in ev6]
print('series deg6:', ev6_ser)
# стационарное m4 при конечном N (rank-1): решить с законом сохранения
# N m4 + N(N-1) m22 = const
exact = dict(eq4={k: str(v) for k, v in eq4.items()}, ev4=[str(e) for e in ev4],
             eq6={k: str(v) for k, v in eq6.items()}, ev6=[str(e) for e in ev6], ev6_series=[str(e) for e in ev6_ser])

# ================= (2) Монте-Карло процесса Каца =================
def kac_mc(Nn, T, n_runs, dt_out=0.25, seed=1, alpha0=0.3):
    rng = np.random.default_rng(seed)
    # начальные данные: i.i.d. из f_alpha (Крук--Ву), выборка отбором
    K = 1 - 2*alpha0
    def sample(n):
        # смесь: (K-a)/K * N(0,K)  +  a/K * [v^2 g_K(v)/K]  (второй компонент: sqrt(K)*sign*|N(0,I_3)|)
        w2 = rng.random(n) < alpha0/K
        g = rng.normal(0, np.sqrt(K), n)
        chi = np.sqrt(np.sum(rng.normal(size=(n, 3))**2, axis=1))*np.sqrt(K)*rng.choice([-1, 1], n)
        return np.where(w2, chi, g)
    nout = int(T/dt_out) + 1
    m4 = np.zeros(nout); m22 = np.zeros(nout)
    for r in range(n_runs):
        v = sample(Nn); t = 0.0; k = 0
        # событий на единицу времени: пары N(N-1)/2 * lam/N = (N-1)/2
        rate = (Nn - 1)/2
        while k < nout:
            # записать при пересечении выходных моментов
            tnext = t + rng.exponential(1/rate)
            while k < nout and k*dt_out <= tnext:
                if k*dt_out >= t:
                    m4[k] += np.mean(v**4); e = v**2; m22[k] += (np.sum(e)**2 - np.sum(e**2))/(Nn*(Nn-1))
                    k += 1
                else: break
            i, j = rng.choice(Nn, 2, replace=False); thv = rng.random()*2*np.pi
            vi, vj_ = v[i], v[j]; v[i] = vi*np.cos(thv) - vj_*np.sin(thv); v[j] = vi*np.sin(thv) + vj_*np.cos(thv)
            t = tnext
    return np.arange(nout)*dt_out, m4/n_runs, m22/n_runs

mc = {}
t0 = time.time()
for Nn, runs in ((8, 8000), (16, 5000), (32, 3000), (64, 2000)):
    tt, m4t, m22t = kac_mc(Nn, 12.0, runs)
    # аппроксимация скорости: log(m4 - m4_inf) линейно; m4_inf из точной формулы 3 + 3(m4(0)-3)/(N+2)
    m4inf = 3 + 3*(m4t[0] - 3)/(Nn + 2)
    y = np.log(np.abs(m4t - m4inf)); msk = (tt >= 0.5) & (tt <= 6.0)
    slope = np.polyfit(tt[msk], y[msk], 1)[0]
    mc[Nn] = dict(t=tt.tolist(), m4=m4t.tolist(), m22=m22t.tolist(), rate=float(-slope), theory=float(0.25*(1 + 2/Nn)), m4inf_theory=float(m4inf))
    print(f'N={Nn}: rate={-slope:.4f} theory={0.25*(1+2/Nn):.4f} m4(0)={m4t[0]:.3f} m4(T)={m4t[-1]:.3f} m4inf_th={m4inf:.3f}  ({time.time()-t0:.0f}s)', flush=True)

# ================= (3) Ляпунов вдоль Крука--Ву в базисе Эрмита =================
NM = 8  # моды 1..8
xg, wg = hermegauss(24); wg = wg/np.sqrt(2*np.pi)
Vg, Wg = np.meshgrid(xg, xg, indexing='ij'); Vg, Wg = Vg.ravel(), Wg.ravel(); WW = np.outer(wg, wg).ravel()
TH = 2*np.pi*(np.arange(32)+0.5)/32
def hn(n, x):
    cc = np.zeros(n+1); cc[n] = 1; return hermeval(x, cc)/np.sqrt(factorial(n))
Hv = np.stack([hn(n, Vg) for n in range(NM+1)]); Hw = np.stack([hn(n, Wg) for n in range(NM+1)])
# Δ[h_n](v,w) усреднённо по θ хранить как (NM+1, n_theta, NV)
DELTA = np.zeros((NM+1, len(TH), len(Vg)))
for it, t in enumerate(TH):
    vp = Vg*np.cos(t) - Wg*np.sin(t); wp = Vg*np.sin(t) + Wg*np.cos(t)
    for n in range(NM+1): DELTA[n, it] = hn(n, vp) + hn(n, wp) - Hv[n] - Hw[n]
def bkw_ratio(a, x):
    K = 1 - 2*a; return np.exp(-x**2/(2*K))/np.sqrt(K)*((K - a)/K + a*x**2/K**2)*np.exp(x**2/2)   # f_alpha/M
def coeffs(a):
    """L_{nk} = <h_n, L_mu h_k>, noise_{nk} = (1/2)∫∫ mu mu Δ[h_n]Δ[h_k], b_{n,jk} = <h_n, (1/2)∫∫ (M h_j)(M h_k) Δ> """
    mu_v = bkw_ratio(a, Vg); mu_w = bkw_ratio(a, Wg); mm = mu_v*mu_w*WW
    noise = 0.5*np.einsum('v,ntv,ktv->nk', mm, DELTA, DELTA)/len(TH)
    # линеаризация: <h_n, L phi> = ∫∫ [mu(v) phi(w) + phi(v) mu(w)] Δ[h_n]/2 ... точная форма: λ∫∫ mu phi Δ[h_n]/2 *2 = ∫∫ mu(v)phi(w)Δ[h_n]
    Lm = np.zeros((NM+1, NM+1))
    for k in range(NM+1):
        Lm[:, k] = np.einsum('v,ntv->n', mu_v*Hw[k]*WW, DELTA)/len(TH)
    # билинейная форма: (1/2)∫∫ G(v,w) Δ[h_n], G = Σ g_jk M h_j ⊗ M h_k
    Bt = 0.5*np.einsum('jv,kv,v,ntv->njk', Hv, Hw, WW, DELTA)/len(TH)
    return Lm, noise, Bt
# проверка в равновесии: Lm диагональ = -gamma_n, noise = 2 gamma_n
Lm0, noise0, Bt0 = coeffs(0.0)
print('L_M diag:', np.round(np.diag(Lm0), 4)); print('noise_M diag:', np.round(np.diag(noise0), 4))
# интегрирование: mu(t) = Крук--Ву (a(t)=a0 e^{-t/8}), G(t): dG/dt = L G + G L^T + noise - (correction для G vs Sigma: используем формулу для Sigma)
# Sigma_{nk}(t) = N Cov(<h_n,f_N>,<h_k,f_N>): dSigma/dt = L Sigma + Sigma L^T + noise
a0, dt, T = 0.30, 0.01, 12.0
Sig = np.zeros((NM+1, NM+1)); Sig[1:, 1:] = 0.0
# начальное: i.i.d. выборка -> Sigma(0) = <h_n h_k>_mu - <h_n><h_k>
mu0 = bkw_ratio(a0, xg)
mom = np.array([np.sum(wg*mu0*hn(n, xg)) for n in range(NM+1)])
Sig = np.array([[np.sum(wg*mu0*hn(n, xg)*hn(k, xg)) for k in range(NM+1)] for n in range(NM+1)]) - np.outer(mom, mom)
ts = [0.0]; sig_hist = [Sig.copy()]; g22_hist = []
t = 0.0
while t < T - 1e-9:
    a = a0*np.exp(-t/8)
    Lm, noise, _ = coeffs(a)
    def rhs(S): return Lm @ S + S @ Lm.T + noise
    k1 = rhs(Sig); k2 = rhs(Sig + dt/2*k1); k3 = rhs(Sig + dt/2*k2); k4 = rhs(Sig + dt*k3)
    Sig = Sig + dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt
    if abs(t/0.5 - round(t/0.5)) < 1e-9:
        ts.append(round(t, 6)); sig_hist.append(Sig.copy())
sig_hist = np.array(sig_hist)
# G = Sigma - (diag part): G_{22} = Sigma_22 - (<h2^2>_mu - <h2>^2); предсказание g22(inf) = sqrt(24) a4(0)/2 (энергетическая мода)
a4_0 = (np.sum(wg*mu0*xg**4) - 3)/np.sqrt(24)
diag22 = [np.sum(wg*bkw_ratio(a0*np.exp(-tt/8), xg)*hn(2, xg)**2) - np.sum(wg*bkw_ratio(a0*np.exp(-tt/8), xg)*hn(2, xg))**2 for tt in ts]
g22 = sig_hist[:, 2, 2] - np.array(diag22)
print('g22(t):', np.round(g22[::4], 4), ' prediction g22(inf)=sqrt(24)a4(0)/2 =', np.sqrt(24)*a4_0/2)
lyap = dict(t=ts, Sig=sig_hist.tolist(), g22=g22.tolist(), g22_pred=float(np.sqrt(24)*a4_0/2),
            L_M=np.diag(Lm0).tolist(), noise_M=np.diag(noise0).tolist())
json.dump(dict(exact=exact, mc={str(k): v for k, v in mc.items()}, lyap=lyap), open('../data/levelC.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
Ns = sorted(mc); ax[0].plot(Ns, [mc[n]['rate'] for n in Ns], 'o', label='Монте-Карло'); nn = np.linspace(6, 70, 100)
ax[0].plot(nn, 0.25*(1+2/nn), 'k-', label=r'$(1+2/N)/4$'); ax[0].axhline(0.25, color='gray', ls=':'); ax[0].set_xlabel('$N$'); ax[0].set_ylabel(r'скорость релаксации $m_4$'); ax[0].legend()
for n in Ns:
    d = mc[n]; ax[1].semilogy(d['t'], np.abs(np.array(d['m4']) - d['m4inf_theory']), label=f'N={n}')
ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'$|m_4-m_4^\infty(N)|$'); ax[1].legend(fontsize=7)
for (n, k) in ((2, 2), (4, 4), (6, 6), (2, 4)):
    ax[2].plot(ts, sig_hist[:, n, k], label=fr'$\Sigma_{{{n}{k}}}$')
ax[2].plot(ts, g22, 'k--', label='$g_{22}$ (двухчаст.)'); ax[2].axhline(lyap['g22_pred'], color='k', ls=':')
ax[2].set_xlabel('$t$'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig_levelC.png'); print('done')
