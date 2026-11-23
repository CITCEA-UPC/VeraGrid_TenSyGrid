import sys,importlib.util,time,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy import sparse
from VeraGridEngine.Simulations.Rms.problems.rms_problem_multilinear import RmsProblemMultilinear
spec=importlib.util.spec_from_file_location('reference',str(Path(__file__).with_name('python_jacobian_reference.py')));ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(18); rows=rng.integers(0,12972,60000);columns=np.repeat(np.arange(20000),3)
s=sparse.csc_matrix((rng.choice([-1.,1.],60000),(rows,columns)),shape=(12972,20000)); v=rng.normal(size=12972);v[::17]=0.
t=time.perf_counter();b=RmsProblemMultilinear._compute_jacobian_sparse_from_S(s,v);first=time.perf_counter()-t
a=ref._compute_jacobian_sparse_from_S(s,v);delta=(a-b).tocoo();error=float(max(abs(delta.data),default=0))
def measure(fn,count):
 times=[]
 for _ in range(count):
  t=time.perf_counter();fn(s,v);times.append(time.perf_counter()-t)
 return float(np.median(times))
old=measure(ref._compute_jacobian_sparse_from_S,10);new=measure(RmsProblemMultilinear._compute_jacobian_sparse_from_S,100)
report=dict(shape=s.shape,nnz=s.nnz,first_numba_call_seconds=first,python_median_seconds=old,numba_median_seconds=new,speedup=old/new,max_absolute_difference=error)
out=Path(__file__).resolve().parent;out.mkdir(exist_ok=True);(out/'kernel_timing.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
