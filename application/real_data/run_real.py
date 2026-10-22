"""Registered pilot on released real polarization observations.

Run locally after acquire.py. Outputs are separate from every earlier study.
"""
from pathlib import Path
import collections,csv,datetime,hashlib,importlib.util,json,platform,time
import numpy as np
import scipy
import scipy.io as sio
from scipy.optimize import minimize
from PIL import Image
P=Path(__file__).resolve().parent;R=P.parents[1]
spec=importlib.util.spec_from_file_location('stokes_core',R/'verification/verify_warm_start.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def save(name,obj):
    (P/name).write_text(json.dumps(c.pack(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stokes(y):
    return np.stack([y.sum(axis=-1)/2,y[...,0]-y[...,2],y[...,1]-y[...,3]],axis=-1)
def sample_static(y,mask,ident,dataset,metadata):
    s=stokes(y);threshold=.05*np.quantile(s[...,0][mask],.99)
    ok=mask&np.all(np.isfinite(y),axis=-1)&np.all(y>=0,axis=-1)&np.all(y<.995,axis=-1)&(s[...,0]>=threshold)
    coords=np.argwhere(ok);rng=np.random.default_rng(np.random.SeedSequence([20261022,ident]))
    selected=rng.choice(len(coords),min(128,len(coords)),replace=False);uniform=coords[selected]
    degree=np.linalg.norm(s[...,1:],axis=-1)/np.maximum(s[...,0],1e-30)
    order=np.argsort(-degree[ok],kind='stable');pool=coords[order[:max(64,int(np.ceil(.01*len(coords))))]]
    high=pool[np.linspace(0,len(pool)-1,min(64,len(pool))).astype(int)]
    out=[]
    for stratum,points in [('uniform',uniform),('high_observed_DoLP',high)]:
        points=points[np.lexsort((points[:,1],points[:,0]))]
        # Static paths are raster-ordered samples, never described as time.
        yy=y[points[:,0],points[:,1]]
        out.append({'dataset':dataset,'sample':metadata['sample'],'stream':f'{dataset}:{metadata["sample"]}:{stratum}','stratum':stratum,'kind':'static_spatial_scan','observations':yy,'coordinates':points,'selection_eligible_pixels':len(coords),'selection_intensity_floor':threshold,'metadata':metadata})
    return out

def mosaic_planes(raw):
    # Exact color/polarizer mapping from the authors' imreadmono2tensor.
    return {
        'R':np.stack([raw[1::4,1::4],raw[0::4,1::4],raw[0::4,0::4],raw[1::4,0::4]],axis=-1),
        'G1':np.stack([raw[1::4,3::4],raw[0::4,3::4],raw[0::4,2::4],raw[1::4,2::4]],axis=-1),
        'G2':np.stack([raw[3::4,1::4],raw[2::4,1::4],raw[2::4,0::4],raw[3::4,0::4]],axis=-1),
        'B':np.stack([raw[3::4,3::4],raw[2::4,3::4],raw[2::4,2::4],raw[3::4,2::4]],axis=-1)}

def inputs():
    streams=[];metadata=[]
    for i,name in enumerate(['box_f','father_christmas_f','horse_l']):
        f=P/'data/DeepSfP/deepSfP_data/indoor'/f'{name}.mat';d=sio.loadmat(f,variable_names=['images','mask']);y=d['images'];mask=d['mask']!=0
        meta={'sample':name,'source':str(f.relative_to(P)),'source_sha256':sha(f),'shape':y.shape,'units':'stored double values, already normalized; no rescaling','ground_truth':'surface normals are not Stokes ground truth; not used'}
        streams+=sample_static(y,mask,i,'DeepSfP',meta);metadata.append(meta)
    for i,(kind,ident) in enumerate([('object','241015104249'),('scene','241021103114')]):
        f=P/f'data/PolarNS/{kind}/{ident}/img_stat.mat';d=sio.loadmat(f);raw=d['img_mean']/4096
        y=np.stack([raw[1::2,1::2],raw[0::2,1::2],raw[0::2,0::2],raw[1::2,0::2]],axis=-1)
        meta={'sample':kind+'_'+ident,'source':str(f.relative_to(P)),'source_sha256':sha(f),'shape':y.shape,'image_count':int(d['image_count'].item()),'units':'img_mean / 4096, matching authors MATLAB code','ground_truth':'averaged observations, not separate noisy frames; no synthetic noise added','raw_variance_median':float(np.median(d['img_variance']))}
        streams+=sample_static(y,np.ones(y.shape[:2],bool),100+i,'PolarNS_means',meta);metadata.append(meta)
    for crop in range(3):
        fs=[P/f'data/PolarBurstSR/test/{crop:07d}/frame_{t:02d}.png' for t in range(14)]
        images=[mosaic_planes(np.array(Image.open(f),dtype=float)/65535) for f in fs]
        for color in ['R','G1','G2','B']:
            frames=np.stack([im[color] for im in images]);y0=frames[0];s=stokes(y0);threshold=.05*np.quantile(s[...,0],.99)
            valid=np.all(np.isfinite(y0),axis=-1)&np.all(y0<.995,axis=-1)&(s[...,0]>=threshold)
            H,W=valid.shape
            grid=np.array([(int((i+.5)*H/4),int((j+.5)*W/4)) for i in range(4) for j in range(4)])
            uniform=np.array([p for p in grid if valid[tuple(p)]],dtype=int).reshape(-1,2)
            coords=np.argwhere(valid);d=np.linalg.norm(s[...,1:],axis=-1)/np.maximum(s[...,0],1e-30)
            ordered=coords[np.argsort(-d[valid],kind='stable')];pool=ordered[:max(8,int(np.ceil(.01*len(coords))))]
            high=pool[np.linspace(0,len(pool)-1,min(8,len(pool))).astype(int)]
            for stratum,points in [('uniform',uniform),('high_observed_DoLP',high)]:
                for track,xy in enumerate(points):
                    yy=frames[:,xy[0],xy[1]]
                    streams.append({'dataset':'PolarBurstSR','sample':f'{crop:07d}_{color}','stream':f'PolarBurstSR:{crop}:{color}:{stratum}:{track}','stratum':stratum,'kind':'real_burst_fixed_sensor_coordinate','observations':yy,'coordinates':np.tile(xy,(14,1)),'selection_eligible_pixels':len(coords),'selection_intensity_floor':threshold,'metadata':{'sample':f'{crop:07d}_{color}','units':'PNG / 65535, matching authors loader','frame_count':14,'motion':'fixed sensor coordinates, no motion compensation or claim of fixed material point','ground_truth':'released high-resolution references not used for aligned reconstruction metrics'}})
        metadata.append({'sample':f'{crop:07d}','dataset':'PolarBurstSR','frames':[{'source':str(f.relative_to(P)),'sha256':sha(f)} for f in fs],'raw_shape':np.array(Image.open(fs[0])).shape})
    return streams,metadata

def diagonal(x,y,need_hessian=True):
    if not np.all(np.isfinite(x)):
        return dict(physical=np.full(3,np.nan),physical_bound=np.nan,physical_accept=False,radius=np.nan,near_fold=False,trigger=False,normalized_radial_curvature=np.nan,recovered_gradient_error=np.nan)
    s,D=c.geometry(x);g=c.A.T@(c.A@s-y);r=float(np.linalg.norm(x[1:]));e=x[1:]/r if r else np.array([1.,0.]);B=c.certificate(s,g,r,e);near=abs(r-1)<=1e-3
    out=dict(physical=s,physical_bound=B,physical_accept=bool(np.isfinite(B) and B<=1e-6),radius=r,near_fold=near,trigger=False,normalized_radial_curvature=None,recovered_gradient_error=None)
    if need_hessian or near:
        z=c.diagnose(x,y);out.update({k:z[k] for k in ['trigger','normalized_radial_curvature','recovered_gradient_error']})
    return out

def oracle_audit(streams):
    out=[]
    for stream in streams[:10]:
        for index in [0,len(stream['observations'])//2]:
            y=stream['observations'][index];s,minimum=c.cone_solution(y)
            def f(z):v=c.A@z-y;return .5*v@v,c.A.T@v
            result=minimize(f,np.r_[max(1e-3,sum(y)/2),0.,0.],jac=True,method='SLSQP',constraints=[{'type':'ineq','fun':lambda z:z[0]-np.linalg.norm(z[1:])}],options={'ftol':1e-13,'maxiter':1000})
            error=abs(result.fun-minimum);assert result.success and error<1e-9,(stream['stream'],error,result.message)
            out.append({'stream':stream['stream'],'index':index,'loss_difference':error,'success':bool(result.success)})
    # Include the first track of every burst crop, irrespective of outcomes.
    for stream in [next(s for s in streams if s['dataset']=='PolarBurstSR' and s['sample'].startswith(f'{i:07d}')) for i in range(3)]:
        for index in [0,7,13]:
            y=stream['observations'][index];s,minimum=c.cone_solution(y)
            def f(z):v=c.A@z-y;return .5*v@v,c.A.T@v
            result=minimize(f,np.r_[max(1e-3,sum(y)/2),0.,0.],jac=True,method='SLSQP',constraints=[{'type':'ineq','fun':lambda z:z[0]-np.linalg.norm(z[1:])}],options={'ftol':1e-13,'maxiter':1000})
            error=abs(result.fun-minimum);assert result.success and error<1e-9
            out.append({'stream':stream['stream'],'index':index,'loss_difference':error,'success':bool(result.success)})
    return out

def summaries(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['dataset'],row['stratum'],row['method'],row['arm'])].append(row)
    out=[]
    for (dataset,stratum,method,arm),rs in groups.items():
        out.append(dict(dataset=dataset,stratum=stratum,method=method,arm=arm,observations=len(rs),streams=len(set(r['stream'] for r in rs)),false_success=sum(r['false_success'] for r in rs),native_failure=sum(not r['success'] for r in rs),inaccurate=sum(r['inaccurate'] for r in rs),interventions=sum(r['intervene'] for r in rs),max_gap=max(r['gap'] for r in rs),rms_distance_to_cone_solution=float(np.sqrt(np.mean([r['distance_to_cone_solution']**2 for r in rs]))),max_distance_to_cone_solution=max(r['distance_to_cone_solution'] for r in rs),physical_false_accept=sum(r['physical_accept'] and r['inaccurate'] for r in rs),local_detected_false_success=sum(r['false_success'] and r['trigger'] for r in rs),local_accurate_success_flagged=sum(r['success'] and not r['inaccurate'] and r['trigger'] for r in rs),function_gradient_pairs=sum(r['fg_calls'] for r in rs),restart_function_gradient_pairs=sum(r['restart_fg_calls'] for r in rs),guarded_curvature_requirements=sum(r['near_fold_initial'] for r in rs) if arm=='curvature_gate' else 0))
    return out

def main():
    assert not (P/'PROTOCOL.json').exists(),'Preserve the registered run.'
    acquisitions={d:sha(P/'sources'/f'{d}_ACQUISITION.json') for d in ['DeepSfP','PolarNS','PolarBurstSR']}
    save('PROTOCOL.json',{'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'objective':'Test fold-related head-coordinate acceptance on a bounded released real-data subset; compare diagnosis and restart with physical-gradient auditing and cold starts. No claim of scene prevalence, denoising/SR superiority or hidden physical gradients.','sources':acquisitions,'selection':'Fixed acquisition list, no solver outcomes inspected. DeepSfP and PolarNS: 128 seeded uniform eligible pixels plus 64 evenly spaced ranks among the top 1 percent observed DoLP, independently per image. Uniform and enriched strata reported separately. PolarBurstSR: each of 4 nominal color planes, fixed 4x4 sensor grid plus 8 enriched positions selected from the first frame only, all 14 frames. Overlap retained and disclosed.','eligibility':'Finite nonnegative observations; all channels below 0.995; nominal S0 at least 5 percent of the sample/plane 99th percentile. DeepSfP additionally requires foreground mask.','units':'DeepSfP stored normalized doubles; PolarNS img_mean/4096; PolarBurstSR PNG/65535. No per-pixel renormalization, clipping, synthetic noise or outcome-based selection.','reference':'Exact ideal four-analyzer weighted cone minimum under W=diag(1,0.5,0.5); nominal linear instrument, not asserted full real-camera calibration or Poisson MLE.','arms':['cold_start','warm_start','curvature_gate','physical_audit'],'methods':['BFGS','L-BFGS-B'],'tolerance':1e-6,'objective_gap_threshold':1e-6,'maxiter':500,'initialization':'First observation of each stream: [log(max(S0_LS,1e-6)),0.2,-0.1]; cold start recomputes this at every observation. Other arms carry their own preceding endpoint, including failures.','static_order':'DeepSfP and PolarNS selected pixels are sorted in raster order, with separate streams per image/stratum. This is a prescribed spatial scan, not time or physical video motion.','temporal_order':'PolarBurstSR follows recorded frame_00..13 at fixed sensor positions; no scene registration or fixed-material-point claim.','curvature_gate':'Existing thresholds abs(r-1)<=1e-3 and c_r/I^2<-1e-7; one restart from best of existing point and radial offsets +/-0.01; same solver/options.','physical_gate':'Reject physical supporting-half-space bound >1e-6; same restart. Physical auditing requires no Hessian.','scope':'Bounded local CPU audit; no training, no new data acquisition, no adaptive grid expansion. Retain zeros and failures; incorporate findings with their scope into paper.','code_sha256':sha(Path(__file__)),'core_sha256':sha(R/'verification/verify_warm_start.py'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}})
    streams,meta=inputs();save('INPUTS.json',streams);save('DATA_SCHEMA.json',meta);save('CONE_ORACLE_AUDIT.json',oracle_audit(streams))
    rows=[];warnings=[];start=time.time()
    for si,stream in enumerate(streams):
        first=stream['observations'][0];x0=np.r_[np.log(max(first.sum()/2,1e-6)),.2,-.1]
        states={(method,arm):x0.copy() for method in ['BFGS','L-BFGS-B'] for arm in ['cold_start','warm_start','curvature_gate','physical_audit']}
        for index,(y,xy) in enumerate(zip(stream['observations'],stream['coordinates'])):
            opt,minimum=c.cone_solution(y)
            for (method,arm),x in states.items():
                if arm=='cold_start':x=np.r_[np.log(max(y.sum()/2,1e-6)),.2,-.1]
                base=c.solve(x,y,method,1e-6);d=diagonal(base['x'],y,need_hessian=arm in ['cold_start','warm_start'])
                intervene=(arm=='curvature_gate' and d['trigger']) or (arm=='physical_audit' and not d['physical_accept'])
                again=c.restart(base,y,method,1e-6) if intervene else None;final=again if again is not None else base
                fd=diagonal(final['x'],y,need_hessian=False);s=fd['physical'];v=c.A@s-y;gap=.5*float(v@v)-minimum
                inaccurate=not np.isfinite(gap) or gap>1e-6
                if np.isfinite(gap) and np.isfinite(fd['physical_bound']):assert fd['physical_bound']+1e-10>=gap
                rows.append(dict(dataset=stream['dataset'],sample=stream['sample'],stream=stream['stream'],kind=stream['kind'],stratum=stream['stratum'],index=index,row=int(xy[0]),column=int(xy[1]),method=method,arm=arm,success=final['success'],status=final['status'],false_success=bool(final['success'] and inaccurate),inaccurate=bool(inaccurate),gap=gap,distance_to_cone_solution=float(np.linalg.norm(s-opt)),physical_accept=fd['physical_accept'],physical_bound=fd['physical_bound'],radius=fd['radius'],near_fold_initial=d['near_fold'],trigger=fd['trigger'],initial_trigger=d['trigger'],intervene=bool(intervene),normalized_radial_curvature=fd['normalized_radial_curvature'],recovered_gradient_error=fd['recovered_gradient_error'],fg_calls=base['counts']['function'],restart_fg_calls=again['counts']['function']+3 if again is not None else 0,nonfinite_calls=base['counts']['nonfinite_calls']+(again['counts']['nonfinite_calls'] if again is not None else 0),initial_x=base['x'],final_x=final['x'],physical=s,cone_solution=opt))
                states[(method,arm)]=np.asarray(final['x'])
                if base['warnings'] or again is not None and again['warnings']:warnings.append({'stream':stream['stream'],'index':index,'method':method,'arm':arm,'initial_warnings':base['warnings'],'restart_warnings':again['warnings'] if again else {}})
        if si%20==0:print(json.dumps({'streams_completed':si+1,'total':len(streams),'elapsed_seconds':round(time.time()-start,2)}),flush=True)
    save('RESULTS.json',{'records':rows,'summary':summaries(rows)});save('WARNINGS.json',warnings)
    counts={d:{'streams':sum(s['dataset']==d for s in streams),'observations':sum(len(s['observations']) for s in streams if s['dataset']==d)} for d in ['DeepSfP','PolarNS_means','PolarBurstSR']}
    summary={'protocol_sha256':sha(P/'PROTOCOL.json'),'inputs_sha256':sha(P/'INPUTS.json'),'results_sha256':sha(P/'RESULTS.json'),'elapsed_seconds':time.time()-start,'counts':counts,'optimizer_fits':len(rows),'summary':summaries(rows),'warnings_records':len(warnings),'no_outcome_based_expansion':True}
    save('SUMMARY.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['summary']}),flush=True)

if __name__=='__main__':main()
