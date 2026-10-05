"""
Однопетлевая поправка к скоростям релаксации из гамильтониана H (модель Каца).
  s_n = -gamma_n + (1/N)[ gamma_n  -  (lam^2/2) Σ_{jk} T_{n,jk} U_{jk,n} / (gamma_j + gamma_k - gamma_n) ]
  T_{n,jk} = ∫∫ MM h_j(v) h_k(w) Δ[h_n](v,w)          (вершина: корреляция -> дрейф)
  U_{jk,n} = ∫∫ MM (h_n(v)+h_n(w)) E_θ[h_j(v') h_k(w')] (вершина: отклонение среднего -> источник корреляций)
Сравнение с точными gamma_n(N): 1/4(1+2/N), 3/8(1+4/N), 29/64 + 3/(32N).
"""
import numpy as np, json
from numpy.polynomial.hermite_e import hermeval, hermegauss
from math import factorial
NM = 10
xg, wg = hermegauss(28); wg = wg/np.sqrt(2*np.pi)
V, W = np.meshgrid(xg, xg, indexing='ij'); V, W = V.ravel(), W.ravel(); WW = np.outer(wg, wg).ravel()
TH = 2*np.pi*(np.arange(32)+0.5)/32
def hn(n, x):
    c = np.zeros(n+1); c[n] = 1; return hermeval(x, c)/np.sqrt(factorial(n))
H = np.stack([hn(n, V) for n in range(NM+1)]); Hw = np.stack([hn(n, W) for n in range(NM+1)])
Hp = np.zeros((NM+1, len(TH), len(V))); Hwp = np.zeros_like(Hp)
for it, t in enumerate(TH):
    vp = V*np.cos(t) - W*np.sin(t); wp = V*np.sin(t) + W*np.cos(t)
    for n in range(NM+1): Hp[n, it] = hn(n, vp); Hwp[n, it] = hn(n, wp)
DELTA = Hp + Hwp - H[:, None, :] - Hw[:, None, :]
gam = np.array([1 - 2*np.mean(np.cos(TH)**n) for n in range(NM+1)]); gam[0] = 0
# вершины
T = np.einsum('jv,kv,v,ntv->njk', H, Hw, WW, DELTA)/len(TH)            # T[n,j,k]
U = np.einsum('v,jtv,ktv->jkt', WW, Hp, Hwp)                             # промежуточно
U = (np.einsum('nv,v,jtv,ktv->jkn', H + Hw, WW, Hp, Hwp))/len(TH)       # U[j,k,n]
out = {}
for n in (4, 6, 8, 10):
    fb = 0.0; terms = []
    for j in range(1, NM+1):
        for k in range(1, NM+1):
            den = gam[j] + gam[k] - gam[n]
            if abs(T[n, j, k]*U[j, k, n]) > 1e-12 and abs(den) > 1e-12:
                fb += T[n, j, k]*U[j, k, n]/den; terms.append((j, k, float(T[n, j, k]), float(U[j, k, n]), float(den)))
    delta = gam[n] + 0.5*fb            # s_n = -gamma_n + delta/N  => gamma_n(N) = gamma_n - delta/N
    exact = {4: 0.5, 6: 1.5, 8: 3/32, 10: 124/128}[n]
    out[n] = dict(gamma=float(gam[n]), delta_oneloop=float(-delta), exact=exact, terms=terms)
    print(f'n={n}: gamma={gam[n]:.5f}  1/N-коэффициент однопетлевой = {-delta:.6f}   точный = {exact:.6f}')
    for tr in terms: print('    (j,k)=(%d,%d) T=%.5f U=%.5f den=%.4f' % tr)
json.dump(out, open('../data/oneloop.json', 'w'), indent=1)
