import sympy as sp, time
src=open('levelC.py').read().split('vars4 = [v1, v2]')[0].split('# ================= (1)')[1]
exec(src.replace('точные конечно-N системы =================',''))
v4, v5 = sp.symbols('v4 v5'); vars10=[v1,v2,v3,v4,v5]
t0=time.time()
phis = {'m10': (v1**10,[v1]), 'm82': (v1**8*v2**2,[v1,v2]), 'm64': (v1**6*v2**4,[v1,v2]),
        'm622': (v1**6*v2**2*v3**2,[v1,v2,v3]), 'm442': (v1**4*v2**4*v3**2,[v1,v2,v3]),
        'm4222': (v1**4*v2**2*v3**2*v4**2,[v1,v2,v3,v4]), 'm22222': (v1**2*v2**2*v3**2*v4**2*v5**2,[v1,v2,v3,v4,v5])}
eq = {k: rate_generator(p, parts, vars10) for k,(p,parts) in phis.items()}
print('eqs done', time.time()-t0, flush=True)
syms=[sp.Symbol(k) for k in phis]
A=sp.Matrix([[sp.diff(eq[k],x) for x in syms] for k in phis])
ev={sp.factor(k):v for k,v in A.eigenvals().items()}
print('eigenvalues deg10:', ev, time.time()-t0)
