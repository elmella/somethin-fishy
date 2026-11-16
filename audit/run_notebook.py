"""Headless execution of the PMP notebook controller and plots.

Usage: python audit/run_notebook.py [updated_sam_fishin.ipynb|igofishing.ipynb]
Historical failed-repair and benchmark artifacts are preserved; outputs use *_pmp.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
name=sys.argv[1] if len(sys.argv)>1 else 'updated_sam_fishin.ipynb'
if name not in ['updated_sam_fishin.ipynb','igofishing.ipynb']:
    raise ValueError('Choose one of the two repaired notebooks.')
book=json.loads((ROOT/'random'/name).read_text())
shown=[]
def display(figure):
    shown.append(figure)
    figure.savefig(ROOT/'audit/results'/name.replace('.ipynb',f'_pmp_plot{len(shown)}.png'),dpi=140)
ns=dict(np=np,solve_ivp=solve_ivp,minimize=minimize,plt=plt,display=display)
for index,cell in enumerate(book['cells']):
    if cell['cell_type']=='code' and 0 <= index <= 25:
        exec(compile(''.join(cell['source']),f'{name}:cell{index}','exec'),ns)
        plt.close('all')
result=dict(notebook=name,J_history=ns['J_history'],
            harvested=float(ns['K']*ns['F0']-ns['F'][:,-1].sum()),
            stop_reason=ns['stop_reason'],method=ns['solution']['method'],
            speed_min=float(ns['U_tensor'][:,:,-1].min()),
            speed_max=float(ns['U_tensor'][:,:,-1].max()),
            verification=ns['verification'],
            note='Saved regularized PMP feedback policy; no unregularized/global-optimum claim.')
path=ROOT/'audit/results'/name.replace('.ipynb','_pmp.json')
path.write_text(json.dumps(result,indent=2))
np.savez(ROOT/'audit/results'/name.replace('.ipynb','_pmp_controls.npz'),
         controls=ns['U_tensor'],grounds=ns['grounds'],ships=ns['ships'])
print(json.dumps(result,indent=2),flush=True)
