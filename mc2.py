import numpy as np, json, time
exec(open('levelC.py').read().split('mc = {}')[0].split('# ================= (2)')[1].replace('Монте-Карло процесса Каца =================',''))
mc = {}; t0=time.time()
for Nn, runs in ((8, 40000), (16, 20000), (32, 10000), (64, 5000)):
    tt, m4t, m22t = kac_mc(Nn, 6.0, runs, dt_out=0.25)
    m4inf = 3 + 3*(m4t[0] - 3)/(Nn + 2)
    y = np.log(np.abs(m4t - m4inf)); msk = tt <= 3.0
    slope = np.polyfit(tt[msk], y[msk], 1)[0]
    exact = m4inf + (m4t[0]-m4inf)*np.exp(-0.25*(1+2/Nn)*tt)
    mc[Nn] = dict(t=tt.tolist(), m4=m4t.tolist(), m22=m22t.tolist(), rate=float(-slope), theory=float(0.25*(1+2/Nn)),
                  m4inf_theory=float(m4inf), exact=exact.tolist(), maxdev=float(np.max(np.abs(m4t-exact))))
    print(f'N={Nn}: rate={-slope:.4f} theory={0.25*(1+2/Nn):.4f} m4(0)={m4t[0]:.3f} m4(T)={m4t[-1]:.3f} exact(T)={exact[-1]:.3f} maxdev={mc[Nn]["maxdev"]:.4f} ({time.time()-t0:.0f}s)', flush=True)
json.dump({str(k): v for k, v in mc.items()}, open('../data/mc2.json','w'), indent=1)
