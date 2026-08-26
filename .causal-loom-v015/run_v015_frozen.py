#!/usr/bin/env python3
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


def git_blob_sha(data: bytes) -> str:
    h=hashlib.sha1(); h.update(f"blob {len(data)}\0".encode()); h.update(data); return h.hexdigest()

def _field(data,name):
    if hasattr(data,name): return np.asarray(getattr(data,name)).squeeze()
    if isinstance(data,np.ndarray) and data.dtype.names and name in data.dtype.names: return np.asarray(data[name]).squeeze()
    raise KeyError(name)

def load_data(path):
    m=loadmat(path,squeeze_me=True,struct_as_record=False)
    if 'data' not in m: raise KeyError(f'no data struct in {path}')
    return m['data']

def filt(x,fs,hz=5.0):
    b,a=butter(2,hz/(fs/2.0)); return filtfilt(b,a,np.asarray(x,dtype=float))

def coordinate(airspeed,rpm,b):
    airspeed=np.asarray(airspeed,dtype=float); rpm=np.asarray(rpm,dtype=float)
    return airspeed*np.power(rpm/5000.0,float(b))

def preprocess_wt(path,p):
    d=load_data(path); D=float(p['physics']['diameter_m']); eff=float(p['physics']['efficiency']); rho=float(p['physics']['rho'])
    air=np.asarray(_field(d,'airspeed'),float); rpm=np.asarray(_field(d,'rpm'),float); voltage=np.asarray(_field(d,'voltage'),float); current=np.asarray(_field(d,'current'),float)
    fs=float(np.asarray(_field(d,'fs')).squeeze())
    power=voltage*current*eff
    rpm=filt(rpm,fs,p['filter']['hz']); power=filt(power,fs,p['filter']['hz'])
    rpm_dot=np.r_[0.0,np.diff(rpm)]*fs
    raw_j_threshold=float(p['selection']['minimum_advance_equivalent'])
    # raw-variable equivalent of J > threshold; no J is passed into coordinate learning/evaluation.
    m=np.isfinite(air)&np.isfinite(rpm)&np.isfinite(power)&(rpm>0)
    m &= power>float(p['selection']['minimum_power_w'])
    m &= np.abs(rpm_dot)<float(p['selection']['maximum_abs_rpm_dot'])
    m &= rpm<float(p['selection']['maximum_rpm'])
    m &= air > raw_j_threshold*(rpm/60.0)*D
    cp=power/(rho*(D**5)*np.power(rpm/60.0,3))
    return {'airspeed':air[m],'rpm':rpm[m],'cp':cp[m],'count_raw':len(air),'count_selected':int(np.sum(m))}

def preprocess_flight(path,p):
    d=load_data(path); D=float(p['physics']['diameter_m']); eff=float(p['physics']['efficiency']); rho=float(p['physics']['rho']); arm=float(p['flight']['motor_arm_m'])
    air=np.asarray(_field(d,'airspeed'),float); rpm=np.asarray(_field(d,'rpm'),float); voltage=np.asarray(_field(d,'voltage'),float); current=np.asarray(_field(d,'current'),float)
    gyrop=np.asarray(_field(d,'gyrop'),float); Vdown=np.asarray(_field(d,'Vdown'),float); Vnorth=np.asarray(_field(d,'Vnorth'),float); Veast=np.asarray(_field(d,'Veast'),float); theta=np.asarray(_field(d,'theta'),float)
    fs=float(np.asarray(_field(d,'fs')).squeeze())
    power=voltage*current*eff; air=air-gyrop*arm; velocity=np.sqrt(Vnorth**2+Veast**2+Vdown**2)
    air=filt(air,fs,p['filter']['hz']); rpm=filt(rpm,fs,p['filter']['hz']); power=filt(power,fs,p['filter']['hz']); Vdown=filt(Vdown,fs,p['filter']['hz']); velocity=filt(velocity,fs,p['filter']['hz']); theta=filt(theta,fs,p['filter']['hz'])
    with np.errstate(invalid='ignore',divide='ignore'):
        gamma=np.arcsin(np.clip(-Vdown/velocity,-1,1)); alpha=theta+np.pi/2.0-gamma
        cp=power/(rho*(D**5)*np.power(rpm/60.0,3))
    m=np.isfinite(air)&np.isfinite(rpm)&np.isfinite(cp)&np.isfinite(alpha)&(rpm>0)
    m &= alpha < math.radians(float(p['flight']['maximum_alpha_deg']))
    m &= power>float(p['flight']['minimum_power_w'])
    m &= rpm<float(p['flight']['maximum_rpm'])
    return {'airspeed':air[m],'rpm':rpm[m],'cp':cp[m],'count_raw':len(air),'count_selected':int(np.sum(m))}

def rank_bins(rpm,n_bins):
    order=np.argsort(np.asarray(rpm),kind='mergesort'); return [x for x in np.array_split(order,int(n_bins)) if len(x)>0]

def smooth_curve(z,y,n_groups):
    z=np.asarray(z,float); y=np.asarray(y,float); o=np.argsort(z,kind='mergesort'); z=z[o]; y=y[o]
    groups=np.array_split(np.arange(len(z)),min(int(n_groups),len(z)))
    xs=[];ys=[]
    for g in groups:
        if len(g): xs.append(float(np.median(z[g])));ys.append(float(np.median(y[g])))
    x=np.asarray(xs); yy=np.asarray(ys); o=np.argsort(x,kind='mergesort'); x=x[o];yy=yy[o]
    ux=[];uy=[]
    for v in np.unique(x):
        m=x==v;ux.append(float(v));uy.append(float(np.mean(yy[m])))
    return np.asarray(ux),np.asarray(uy)

def pair_collapse(a,b,min_overlap_points,den_floor):
    x1,y1=a;x2,y2=b; lo=max(float(x1.min()),float(x2.min()));hi=min(float(x1.max()),float(x2.max()))
    m=(x1>=lo)&(x1<=hi)
    if int(m.sum())<int(min_overlap_points): return None
    pred=np.interp(x1[m],x2,y2); den=max(float(np.ptp(np.r_[y1[m],pred])),float(den_floor))
    return float(np.sqrt(np.mean((y1[m]-pred)**2))/den)

def collapse_metric(air,rpm,cp,b,p):
    vals=[]; bins=rank_bins(rpm,p['wt_evaluation']['rpm_rank_bins']); curves=[]
    for idx in bins:
        z=coordinate(air[idx],rpm[idx],b); curves.append(smooth_curve(z,cp[idx],p['wt_evaluation']['curve_groups']))
    for i in range(len(curves)):
        for j in range(i+1,len(curves)):
            v=pair_collapse(curves[i],curves[j],p['wt_evaluation']['minimum_overlap_curve_points'],p['wt_evaluation']['cp_range_floor'])
            if v is not None and math.isfinite(v): vals.append(v)
    if not vals: return {'median':float('inf'),'pair_count':0,'values':[]}
    return {'median':float(np.median(vals)),'pair_count':len(vals),'values':vals}

def poly_features(z,degree,zmin=None,zmax=None):
    z=np.asarray(z,float)
    if zmin is None: zmin=float(np.min(z))
    if zmax is None: zmax=float(np.max(z))
    if not zmax>zmin: raise ValueError('degenerate coordinate support')
    u=2*(z-zmin)/(zmax-zmin)-1
    X=np.polynomial.chebyshev.chebvander(u,int(degree))
    return X,float(zmin),float(zmax)

def bin_cv(air,rpm,cp,b,p):
    bins=rank_bins(rpm,p['wt_evaluation']['rpm_rank_bins']); errs=[]
    for k,test in enumerate(bins):
        train=np.concatenate([bb for i,bb in enumerate(bins) if i!=k])
        ztr=coordinate(air[train],rpm[train],b); zte=coordinate(air[test],rpm[test],b)
        X,zmin,zmax=poly_features(ztr,p['wt_evaluation']['polynomial_degree'])
        beta=np.linalg.lstsq(X,cp[train],rcond=None)[0]
        m=(zte>=zmin)&(zte<=zmax)
        if int(m.sum())<p['wt_evaluation']['minimum_test_points']: continue
        Xt,_,_=poly_features(zte[m],p['wt_evaluation']['polynomial_degree'],zmin,zmax)
        den=max(float(np.ptp(cp[test][m])),p['wt_evaluation']['cp_range_floor'])
        errs.append(float(np.sqrt(np.mean((Xt@beta-cp[test][m])**2))/den))
    return {'median':float(np.median(errs)) if errs else float('inf'),'fold_count':len(errs),'values':errs}

def fit_wt_predict_flight(wt,fl,b,p):
    zw=coordinate(wt['airspeed'],wt['rpm'],b); zf=coordinate(fl['airspeed'],fl['rpm'],b)
    X,zmin,zmax=poly_features(zw,p['cross_modality']['polynomial_degree']); beta=np.linalg.lstsq(X,wt['cp'],rcond=None)[0]
    return beta,zmin,zmax,zf

def flight_transfer(wt,fl,b_corr,b_can,p):
    bc,lc,hc,zfc=fit_wt_predict_flight(wt,fl,b_corr,p); bj,lj,hj,zfj=fit_wt_predict_flight(wt,fl,b_can,p)
    m=(zfc>=lc)&(zfc<=hc)&(zfj>=lj)&(zfj<=hj)&np.isfinite(fl['cp'])
    if int(m.sum())<p['cross_modality']['minimum_scored_points']: return {'valid':False,'scored_points':int(m.sum())}
    Xc,_,_=poly_features(zfc[m],p['cross_modality']['polynomial_degree'],lc,hc); Xj,_,_=poly_features(zfj[m],p['cross_modality']['polynomial_degree'],lj,hj)
    y=fl['cp'][m]; den=max(float(np.ptp(y)),p['cross_modality']['cp_range_floor'])
    ec=float(np.sqrt(np.mean((Xc@bc-y)**2))/den); ej=float(np.sqrt(np.mean((Xj@bj-y)**2))/den)
    return {'valid':True,'scored_points':int(m.sum()),'corrected_nrmse':ec,'canonical_nrmse':ej,'ratio':ec/ej if ej>0 else float('inf')}

def self_test():
    # Coordinate exponent should undo a known RPM-dependent source stretch.
    j=np.linspace(.2,.8,200); r1=4000.;r2=8000.; D=.2
    v1=j*(r1/60)*D; v2=j*(r2/60)*D
    z1=coordinate(v1,np.full_like(v1,r1),-1.2);z2=coordinate(v2,np.full_like(v2,r2),-1.2)
    f=lambda z:.12-.03*z-.002*z*z
    y1=f(z1);y2=f(z2)
    c=pair_collapse(smooth_curve(z1,y1,80),smooth_curve(z2,y2,80),20,.001)
    assert c is not None and c<1e-4
    assert git_blob_sha(b'hello\n')=='ce013625030ba8dba906f756967f9e9ca394464a'
    print('SELF_TEST_PASS')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol');ap.add_argument('--wt');ap.add_argument('--flight');ap.add_argument('--output-dir');ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test: self_test(); return 0
    p=json.load(open(a.protocol)); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for key,path in [('wt',Path(a.wt)),('flight',Path(a.flight))]:
        expected=p['source_files'][key]['blob_sha']; actual=git_blob_sha(path.read_bytes())
        if actual!=expected: raise RuntimeError(f'{key} blob mismatch {actual} != {expected}')
    wt=preprocess_wt(Path(a.wt),p); fl=preprocess_flight(Path(a.flight),p)
    bc=float(p['coordinate']['frozen_rpm_exponent']); bj=float(p['coordinate']['canonical_rpm_exponent'])
    cc=collapse_metric(wt['airspeed'],wt['rpm'],wt['cp'],bc,p); cj=collapse_metric(wt['airspeed'],wt['rpm'],wt['cp'],bj,p)
    cv_c=bin_cv(wt['airspeed'],wt['rpm'],wt['cp'],bc,p); cv_j=bin_cv(wt['airspeed'],wt['rpm'],wt['cp'],bj,p)
    n=min(len(cc['values']),len(cj['values'])); win=float(np.mean(np.asarray(cc['values'][:n])<np.asarray(cj['values'][:n]))) if n else 0.0
    flight=flight_transfer(wt,fl,bc,bj,p)
    scan=[]
    for b in np.arange(p['diagnostic_scan']['min'],p['diagnostic_scan']['max']+1e-12,p['diagnostic_scan']['step']):
        m=collapse_metric(wt['airspeed'],wt['rpm'],wt['cp'],float(b),p);scan.append([float(b),m['median']])
    finite=[x for x in scan if math.isfinite(x[1])]; best=min(finite,key=lambda x:x[1]) if finite else [None,None]
    gates={
      'minimum_wt_selected_points': {'value':wt['count_selected'],'threshold':p['primary_gates']['minimum_wt_selected_points'],'pass':wt['count_selected']>=p['primary_gates']['minimum_wt_selected_points']},
      'minimum_pair_count': {'value':cc['pair_count'],'threshold':p['primary_gates']['minimum_pair_count'],'pass':cc['pair_count']>=p['primary_gates']['minimum_pair_count']},
      'collapse_ratio': {'value':cc['median']/cj['median'] if cj['median']>0 else float('inf'),'threshold':p['primary_gates']['collapse_corrected_vs_canonical_ratio_lte'],'pass':cc['median']/cj['median']<=p['primary_gates']['collapse_corrected_vs_canonical_ratio_lte']},
      'collapse_win_fraction': {'value':win,'threshold':p['primary_gates']['collapse_corrected_win_fraction_gte'],'pass':win>=p['primary_gates']['collapse_corrected_win_fraction_gte']},
      'wt_bin_cv_ratio': {'value':cv_c['median']/cv_j['median'] if cv_j['median']>0 else float('inf'),'threshold':p['primary_gates']['wt_bin_cv_corrected_vs_canonical_ratio_lte'],'pass':cv_c['median']/cv_j['median']<=p['primary_gates']['wt_bin_cv_corrected_vs_canonical_ratio_lte']},
    }
    strong={
      'flight_transfer_valid': {'value':flight.get('valid',False),'threshold':True,'pass':bool(flight.get('valid',False))},
      'flight_transfer_ratio': {'value':flight.get('ratio',float('inf')),'threshold':p['strong_claim_gates']['wt_to_flight_corrected_vs_canonical_ratio_lte'],'pass':bool(flight.get('valid')) and flight['ratio']<=p['strong_claim_gates']['wt_to_flight_corrected_vs_canonical_ratio_lte']},
      'external_optimum_near_frozen': {'value':best[0],'threshold':p['strong_claim_gates']['posthoc_best_b_abs_distance_lte'],'pass':best[0] is not None and abs(best[0]-bc)<=p['strong_claim_gates']['posthoc_best_b_abs_distance_lte']},
    }
    res={'protocol_id':p['protocol_id'],'primary_pass':all(g['pass'] for g in gates.values()),'strong_claim_pass':all(g['pass'] for g in strong.values()),'wt':{'counts':{'raw':wt['count_raw'],'selected':wt['count_selected']},'corrected_collapse':cc,'canonical_collapse':cj,'corrected_bin_cv':cv_c,'canonical_bin_cv':cv_j,'pairwise_win_fraction':win},'flight':flight,'diagnostic_scan':{'best_b':best[0],'best_collapse':best[1],'grid':scan},'gates':gates,'strong_claim_gates':strong}
    (out/'results.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    verdict=['# Causal Loom v0.15 frozen external result','',f"Primary pass: **{res['primary_pass']}**",f"Strong cross-modality pass: **{res['strong_claim_pass']}**",'',f"WT corrected/canonical collapse ratio: {gates['collapse_ratio']['value']:.6f}",f"WT corrected/canonical bin-CV ratio: {gates['wt_bin_cv_ratio']['value']:.6f}",f"WT pairwise win fraction: {win:.6f}",f"WT->flight ratio: {flight.get('ratio',float('nan')):.6f}",f"Post-hoc WT best b: {best[0]}"]
    (out/'verdict.md').write_text('\n'.join(verdict)+'\n')
    print(json.dumps({'primary_pass':res['primary_pass'],'strong_claim_pass':res['strong_claim_pass'],'gates':gates,'strong':strong,'wt_selected':wt['count_selected'],'flight':flight,'best_b':best[0]},indent=2))
    return 0 if res['primary_pass'] else 2
if __name__=='__main__': raise SystemExit(main())
