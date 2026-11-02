"""Independent weight, Hessian, and cone-solver checks of saved training runs."""
from pathlib import Path
import argparse
import json

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=['head','cone'],required=True)
    args = parser.parse_args()
    report = json.loads((HERE/'results/trained_fold_diagnostic.json').read_text())
    if args.part == 'head':
        import torch
        from trained_fold_core import FoldMLP
        from trained_fold_diagnostic import diagnose
        torch.set_num_threads(1)
        max_curvature, max_gradient, count = 0., 0., 0
        for run in report['runs']:
            stem=f"{run['dataset']}_{run['initialization']}_{run['seed']}"
            data=dict(np.load(HERE/'results/trained_fold'/(stem+'.npz'),allow_pickle=False))
            model=FoldMLP(run['initialization'])
            model.load_state_dict({k.removeprefix('weight__'):torch.from_numpy(v.copy())
                                   for k,v in data.items() if k.startswith('weight__')})
            np.testing.assert_array_equal(model(torch.from_numpy(data['y']))[0].detach().numpy(),data['s'])
            values=diagnose(data['I'],data['w'],data['y'],data['A'])['arrays']
            for index in [0,333,666,1023]:
                intensity=torch.tensor(float(data['I'][index]),dtype=torch.float64)
                w=torch.tensor(data['w'][index].astype(float),requires_grad=True)
                A=torch.tensor(data['A'].astype(float));y=torch.tensor(data['y'][index].astype(float))
                state=intensity*torch.cat((torch.ones(1,dtype=w.dtype),2*w/(1+(w*w).sum())))
                loss=.5*((A@state-y)**2).sum()
                j,g=torch.autograd.grad(loss,(w,state),create_graph=True)
                e=w.detach()/torch.linalg.vector_norm(w.detach())
                v=torch.autograd.grad(j@e,w)[0]
                max_curvature=max(max_curvature,abs(float((v@e).detach())-values['radial_curvature'][index]))
                max_gradient=max(max_gradient,float(np.max(abs(g.detach().numpy()-values['physical_gradient'][index]))))
                count+=1
        A=np.array([[1,1,0],[1,0,1],[1,-1,0],[1,0,-1]])*.5
        control=diagnose(np.array([1.]),np.array([[1.,0.]]),np.array([[1.25,.5,0.]])@A.T,A)
        assert control['counts']['0.001']['stationary_negative']==1
        assert control['counts']['0.001']['max_trial_drop']>0
        assert max_curvature<1e-12 and max_gradient<1e-12
        result=dict(weights_replayed=len(report['runs']),head_HVP_checks=count,
                    max_curvature_discrepancy=max_curvature,max_gradient_discrepancy=max_gradient,
                    positive_detector_control_passed=True)
    else:
        from scipy.optimize import minimize
        maximum=0.
        for run in report['runs']:
            stem=f"{run['dataset']}_{run['initialization']}_{run['seed']}"
            data=np.load(HERE/'results/trained_fold'/(stem+'.npz'),allow_pickle=False)
            diag=np.load(HERE/'results/trained_fold'/(stem+'_diagnostic.npz'),allow_pickle=False)
            A=data['A'].astype(float);y=data['y'][333].astype(float)
            I=float(data['I'][333]);w=data['w'][333].astype(float)
            state=I*np.r_[1.,2*w/(1+w@w)]
            loss=.5*np.sum((A@state-y)**2)
            solved=minimize(lambda u:.5*np.sum((A@u-y)**2),state,
                jac=lambda u:A.T@(A@u-y),method='SLSQP',
                constraints=[{'type':'ineq','fun':lambda u:u[0]-np.linalg.norm(u[1:])}],
                options={'ftol':1e-13,'maxiter':500})
            assert solved.success
            maximum=max(maximum,abs(loss-solved.fun-diag['gap'][333]))
        assert maximum<1e-10
        result=dict(independent_SLSQP_checks=len(report['runs']),max_gap_discrepancy=float(maximum))
    (HERE/'results'/f'trained_fold_{args.part}_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
