"""Correct the alternative-map comparison to use the same physical initial state.

The original raw-coordinate control remains immutable in results.json.
No physical conditions, seeds, tolerance or input observations are added.
"""
from pathlib import Path
import datetime,hashlib,importlib.util,json,time
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fresnel_pilot',HERE/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
protocol=HERE/'MATCHED_CONTROL_PROTOCOL.json'
assert not protocol.exists(),'Preserve this correction audit.'
x_fold=np.array([0.,.2,-.1]);s0,_=m.c.geometry(x_fold);p=s0[1:]/s0[0]
x0=np.r_[np.log(s0[0]),p/np.sqrt(1-p@p)]
assert np.max(np.abs(m.mono_geometry(x0)[0]-s0))<1e-15
m.save(protocol.name,{'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'The original control uses the same raw coordinate, which gives a different physical initial state under the monotone map. This additional control matches the physical initial state. Retain both controls.','grid':'Exactly the original 984 observations; no new conditions or seeds','initial_monotone_coordinate':x0,'physical_initial_state':s0,'source_inputs_sha256':m.digest(HERE/'inputs.json'),'code_sha256':m.digest(Path(__file__)),'other_options':'BFGS, gtol=1e-6, maxiter=500; previous monotone raw state carried forward; unchanged objective and cone gap threshold'})
inputs=json.loads((HERE/'inputs.json').read_text());records=[];start=time.time()
for inp in inputs:
    x=x0.copy()
    for frame,(y,truth) in enumerate(zip(inp['observations'],inp['truth'])):
        y=np.array(y);opt,minimum=m.c.cone_solution(y)
        final=m.assess(m.mono_solve(x,y),y,minimum,True);x=np.array(final['x'])
        records.append(dict(sequence=inp['sequence'],frame=frame,truth=truth,final=final))
summary={'frames':len(records),'false_success':sum(r['final']['false_success'] for r in records),'solver_nonacceptances':sum(not r['final']['success'] for r in records),'max_gap':max(r['final']['gap'] for r in records),'stokes_RMSE':float(np.sqrt(np.mean([(np.array(r['final']['physical'])-r['truth'])**2 for r in records]))),'elapsed_seconds':time.time()-start,'function_gradient_pairs':sum(r['final']['counts']['function'] for r in records)}
m.save('matched_monotone_control.json',{'protocol_sha256':m.digest(protocol),'summary':summary,'records':records});print(json.dumps(summary))
