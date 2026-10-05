import sympy as sp, time
src=open('levelC.py').read().split('vars4 = [v1, v2]')[0].split('# ================= (1)')[1]
exec(src.replace('точные конечно-N системы =================',''))
v4 = sp.Symbol('v4'); vars8=[v1,v2,v3,v4]
t0=time.time()
phis = {'m8': (v1**8,[v1]), 'm62': (v1**6*v2**2,[v1,v2]), 'm44': (v1**4*v2**4,[v1,v2]),
        'm422': (v1**4*v2**2*v3**2,[v1,v2,v3]), 'm2222': (v1**2*v2**2*v3**2*v4**2,[v1,v2,v3,v4])}
eq8 = {k: rate_generator(p, parts, vars8) for k,(p,parts) in phis.items()}
print(eq8, time.time()-t0, flush=True)
syms=[sp.Symbol(k) for k in phis]
A=sp.Matrix([[sp.diff(eq8[k],x) for x in syms] for k in phis])
ev={sp.factor(k):v for k,v in A.eigenvals().items()}
print('eigenvalues deg8:', ev, time.time()-t0)
