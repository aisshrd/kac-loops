import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 600})
mc = json.load(open('../data/mc2.json')); L = json.load(open('../data/levelC.json'))['lyap']
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
Ns = sorted(int(k) for k in mc); ax[0].plot(Ns, [mc[str(n)]['rate'] for n in Ns], 'o', label='Монте-Карло'); nn = np.linspace(6, 70, 100)
ax[0].plot(nn, 0.25*(1+2/nn), 'k-', label=r'$\gamma_4(N)=(1+2/N)/4$'); ax[0].axhline(0.25, color='gray', ls=':'); ax[0].set_xlabel('$N$'); ax[0].set_ylabel('скорость релаксации $m_4$'); ax[0].legend(fontsize=8)
for n in Ns:
    d = mc[str(n)]; ax[1].plot(d['t'], d['m4'], 'o', ms=3, label=f'N={n}'); ax[1].plot(d['t'], d['exact'], 'k-', lw=0.8)
ax[1].set_xlabel('$t$'); ax[1].set_ylabel('$m_4(t)$'); ax[1].legend(fontsize=7)
ts = np.array(L['t']); S = np.array(L['Sig'])
for (n, k) in ((2, 2), (4, 4), (6, 6), (2, 4)):
    ax[2].plot(ts, S[:, n, k], label=fr'$\Sigma_{{{n}{k}}}$')
ax[2].plot(ts, L['g22'], 'k--', label='$g_{22}$'); ax[2].axhline(L['g22_pred'], color='k', ls=':')
ax[2].set_xlabel('$t$'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig_levelC.png'); print('ok')
