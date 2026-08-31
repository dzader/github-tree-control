#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

EPS=1e-12
OUTPUTS=("ct","cp")


def load_shared(path:Path):
    spec=importlib.util.spec_from_file_location("v019_shared_lowrank",path)
    if spec is None or spec.loader is None:raise RuntimeError("cannot load shared module")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def geom(f):
    ld=f.log_d;pr=f.pitch_ratio
    return np.asarray([1.,ld,pr,ld*pr,ld*ld,pr*pr])


@dataclass
class LowRankAtlas:
    bounds:Dict[str,float]
    coeffs:Dict[str,np.ndarray] # p x 2: base, tangent
    c_prior_beta:np.ndarray
    output_floors:Dict[str,float]
    c_lo:float=-1.6
    c_hi:float=1.6
    prior_lambda:float=0.025

    def modes(self,m,f,output):
        B=m.tensor_cheb(f.z_fixed,f.r,self.bounds)
        H=B@self.coeffs[output]
        return H[:,0],H[:,1]

    def prior(self,f):
        return float(np.clip(geom(f)@self.c_prior_beta,self.c_lo,self.c_hi))


def normalized_target(f,output):
    y=f.y(output);center=float(np.median(y));scale=max(float(np.std(y)),float(np.percentile(y,90)-np.percentile(y,10))/2.563,1e-4)
    return (y-center)/scale


def projective_corr(a,b):
    A=np.column_stack([np.ones(len(a)),a]);beta=np.linalg.lstsq(A,b,rcond=None)[0];br=b-A@beta
    ar=a-np.mean(a)
    if np.std(ar)<1e-12 or np.std(br)<1e-12:return 0.0
    return float(abs(np.corrcoef(ar,br)[0,1]))


def orthogonalize_mode(m,train,bounds,C0,C1):
    B=np.vstack([m.tensor_cheb(f.z_fixed,f.r,bounds) for f in train])
    h0=B@C0;h1=B@C1
    X=np.column_stack([np.ones(len(h0)),h0]);beta=np.linalg.lstsq(X,h1,rcond=None)[0]
    constant=np.zeros_like(C0);constant[0]=1.0
    C1=C1-beta[0]*constant-beta[1]*C0
    h1=B@C1;scale=max(float(np.std(h1)),1e-8)
    return C1/scale


def fit_affine_curve(m,h,y):
    return m.fit_affine(h,y,2e-5)


def family_c_score(m,atlas,f,c,indices=None):
    if indices is None:indices=np.arange(len(f.j))
    parts=[]
    for output in OUTPUTS:
        h0,h1=atlas.modes(m,f,output);h=h0+c*h1
        a,b=fit_affine_curve(m,h[indices],f.y(output)[indices])
        den=max(float(np.ptp(f.y(output)[indices])),atlas.output_floors[output])
        parts.append(float(np.sqrt(np.mean((a+b*h[indices]-f.y(output)[indices])**2))/den))
    return math.sqrt(max(parts[0],EPS)*max(parts[1],EPS))


def estimate_c_full(m,atlas,f):
    grid=np.linspace(atlas.c_lo,atlas.c_hi,81);scores=[family_c_score(m,atlas,f,float(c)) for c in grid]
    return float(grid[int(np.argmin(scores))])


def initial_coeffs(m,train,bounds):
    coeffs={}
    for output in OUTPUTS:
        blocks=[];ys=[];ws=[]
        for f in train:
            blocks.append(m.tensor_cheb(f.z_fixed,f.r,bounds));ys.append(normalized_target(f,output));ws.append(np.full(len(f.j),1/len(f.j)))
        B=np.vstack(blocks);y=np.concatenate(ys);w=np.concatenate(ws)
        C0=m.ridge_solve(B,y,4e-3,w)
        # Initialize the deformation as the z derivative of the learned base mechanism.
        all_z=np.concatenate([f.z_fixed for f in train]);all_r=np.concatenate([f.r for f in train]);eps=1e-3
        hp=m.tensor_cheb(all_z+eps,all_r,bounds)@C0;hm=m.tensor_cheb(all_z-eps,all_r,bounds)@C0
        deriv=(hp-hm)/(2*eps)
        B_all=m.tensor_cheb(all_z,all_r,bounds)
        C1=m.ridge_solve(B_all,deriv,1e-3)
        C1=orthogonalize_mode(m,train,bounds,C0,C1)
        coeffs[output]=np.column_stack([C0,C1])
    return coeffs


def update_coeffs(m,train,atlas,c_values):
    result={}
    for output in OUTPUTS:
        Xs=[];ys=[];ws=[]
        for f in train:
            h0,h1=atlas.modes(m,f,output);h=h0+c_values[f.name]*h1
            a,b=fit_affine_curve(m,h,f.y(output))
            if abs(b)<1e-7:continue
            target=(f.y(output)-a)/b
            B=m.tensor_cheb(f.z_fixed,f.r,atlas.bounds);c=c_values[f.name]
            Xs.append(np.hstack([B,c*B]));ys.append(target);ws.append(np.full(len(f.j),1/len(f.j)))
        beta=m.ridge_solve(np.vstack(Xs),np.concatenate(ys),8e-3,np.concatenate(ws))
        p=beta.shape[0]//2;C0=beta[:p];C1=orthogonalize_mode(m,train,atlas.bounds,C0,beta[p:])
        result[output]=np.column_stack([C0,C1])
    return result


def fit_atlas(m,train):
    z=np.concatenate([f.z_fixed for f in train]);r=np.concatenate([f.r for f in train])
    bounds={"z_lo":float(np.min(z)-0.12),"z_hi":float(np.max(z)+0.12),"r_lo":float(np.min(r)),"r_hi":float(np.max(r))}
    floors={o:max(float(np.median([np.ptp(f.y(o)) for f in train]))*.15,1e-4) for o in OUTPUTS}
    atlas=LowRankAtlas(bounds,initial_coeffs(m,train,bounds),np.zeros(6),floors)
    c_values={f.name:0.0 for f in train}
    for _ in range(5):
        c_values={f.name:estimate_c_full(m,atlas,f) for f in train}
        atlas.coeffs=update_coeffs(m,train,atlas,c_values)
    G=np.vstack([geom(f) for f in train]);c=np.asarray([c_values[f.name] for f in train])
    atlas.c_prior_beta=m.ridge_solve(G,c,.35)
    return atlas,c_values


def atlas_predict(m,atlas,f,probes):
    prior=atlas.prior(f);grid=np.unique(np.clip(np.r_[np.linspace(prior-1.0,prior+1.0,81),prior,0.0],atlas.c_lo,atlas.c_hi))
    best=(float("inf"),prior,{})
    for c in grid:
        params={};parts=[]
        for output in OUTPUTS:
            h0,h1=atlas.modes(m,f,output);h=h0+c*h1;a,b=fit_affine_curve(m,h[probes],f.y(output)[probes]);params[output]=(a,b)
            parts.append(float(np.sqrt(np.mean((a+b*h[probes]-f.y(output)[probes])**2))/atlas.output_floors[output]))
        score=math.sqrt(max(parts[0],EPS)*max(parts[1],EPS))+atlas.prior_lambda*((c-prior)/.8)**2
        if score<best[0]:best=(score,float(c),params)
    _,c,params=best;pred={}
    for output in OUTPUTS:
        h0,h1=atlas.modes(m,f,output);a,b=params[output];pred[output]=a+b*(h0+c*h1)
    return pred,c


def choose_blend(m,atlas,f,probes):
    rows=[]
    for held in probes:
        sub=probes[probes!=held];ap,_=atlas_predict(m,atlas,f,sub);fp=m.fixed_cubic_predict(f,sub,"fixed")
        rows.append((int(held),ap,fp))
    best=(float("inf"),0.)
    for alpha in np.linspace(0,1,11):
        e={o:[] for o in OUTPUTS}
        for held,ap,fp in rows:
            for o in OUTPUTS:e[o].append((alpha*ap[o][held]+(1-alpha)*fp[o][held]-f.y(o)[held])/atlas.output_floors[o])
        score=math.sqrt(max(float(np.sqrt(np.mean(np.square(e['ct'])))),EPS)*max(float(np.sqrt(np.mean(np.square(e['cp'])))),EPS))+.008*alpha
        if score<best[0]:best=(score,float(alpha))
    return best[1]


def safe_predict(m,atlas,f,probes):
    ap,c=atlas_predict(m,atlas,f,probes);fp=m.fixed_cubic_predict(f,probes,"fixed");alpha=choose_blend(m,atlas,f,probes)
    return {o:alpha*ap[o]+(1-alpha)*fp[o] for o in OUTPUTS},c,alpha


def tangent_diagnostic(m,atlas,train):
    rmed=float(np.median(np.concatenate([f.r for f in train])));z=np.linspace(atlas.bounds['z_lo']+.05,atlas.bounds['z_hi']-.05,300);r=np.full_like(z,rmed);eps=1e-4
    out={}
    for o in OUTPUTS:
        B=m.tensor_cheb(z,r,atlas.bounds);h0=B@atlas.coeffs[o][:,0];h1=B@atlas.coeffs[o][:,1]
        hp=m.tensor_cheb(z+eps,r,atlas.bounds)@atlas.coeffs[o][:,0];hm=m.tensor_cheb(z-eps,r,atlas.bounds)@atlas.coeffs[o][:,0];d=(hp-hm)/(2*eps)
        out[o]=projective_corr(d,h1)
    return out


def aggregate(records,name):
    v=np.asarray([r[f'{name}_joint'] for r in records],float);fixed=np.asarray([r['fixed_joint'] for r in records],float);ratio=v/np.maximum(fixed,EPS)
    return {"median_joint":float(np.median(v)),"geometric_mean_joint":float(np.exp(np.mean(np.log(np.maximum(v,EPS))))),"median_ratio_vs_fixed":float(np.median(ratio)),"geometric_mean_ratio_vs_fixed":float(np.exp(np.mean(np.log(np.maximum(ratio,EPS))))),"win_fraction_vs_fixed":float(np.mean(ratio<1)),"p90_ratio_vs_fixed":float(np.quantile(ratio,.9)),"worst_ratio_vs_fixed":float(np.max(ratio))}


def run(m,families):
    records=[];fold_diags=[]
    for fold in range(5):
        train=[f for f in families if m.stable_fold(f.name)!=fold];test=[f for f in families if m.stable_fold(f.name)==fold]
        if not test:continue
        atlas,_=fit_atlas(m,train);controls=m.fit_global_controls(train);fold_diags.append(tangent_diagnostic(m,atlas,train))
        for f in test:
            probes=m.select_probes(f,5);scored=np.asarray([i for i in range(len(f.j)) if i not in set(probes.tolist())],int)
            methods={"fixed":m.fixed_cubic_predict(f,probes,"fixed"),"plain_j":m.fixed_cubic_predict(f,probes,"j")}
            safe,c,alpha=safe_predict(m,atlas,f,probes);raw,_=atlas_predict(m,atlas,f,probes);methods['safe_atlas']=safe;methods['raw_atlas']=raw
            for name in controls.models:methods[name]=m.calibrated_global_predict(controls,name,f,probes)
            row={"family":f.name,"fold":fold,"diameter_in":f.diameter_in,"pitch_in":f.pitch_in,"points":len(f.j),"atlas_c":c,"atlas_weight":alpha}
            for name,pred in methods.items():
                joint,errs=m.family_joint_error(f,pred,scored,atlas.output_floors);row[f'{name}_joint']=joint;row[f'{name}_ct']=errs['ct'];row[f'{name}_cp']=errs['cp']
            records.append(row)
    names=['fixed','plain_j','safe_atlas','raw_atlas','direct_poly','extra_trees','hist_gb','mlp'];methods={n:aggregate(records,n) for n in names};a=methods['safe_atlas'];best=min(methods[n]['geometric_mean_joint'] for n in ('direct_poly','extra_trees','hist_gb','mlp'))
    summary={"protocol":"five family-disjoint folds; exactly five output-blind maximin probes; matched cubic and learned controls","family_count":len(records),"methods":methods,"tangent_mode_projective_correlation":{"ct_median":float(np.median([x['ct'] for x in fold_diags])),"cp_median":float(np.median([x['cp'] for x in fold_diags])),"per_fold":fold_diags},"atlas_diagnostics":{"median_c":float(np.median([r['atlas_c'] for r in records])),"median_weight":float(np.median([r['atlas_weight'] for r in records])),"nonzero_weight_fraction":float(np.mean([r['atlas_weight']>0 for r in records]))}}
    summary['development_gate']={"pass":bool(a['geometric_mean_ratio_vs_fixed']<=.92 and a['win_fraction_vs_fixed']>=.65 and a['worst_ratio_vs_fixed']<=1.8 and a['geometric_mean_joint']<=.90*best and summary['tangent_mode_projective_correlation']['ct_median']>=.75 and summary['tangent_mode_projective_correlation']['cp_median']>=.75),"requirements":{"geometric_mean_ratio_vs_fixed_lte":.92,"win_fraction_vs_fixed_gte":.65,"worst_ratio_vs_fixed_lte":1.8,"ratio_vs_best_learned_control_lte":.90,"tangent_correlation_each_gte":.75},"best_learned_control_geometric_mean_joint":best,"atlas_ratio_vs_best_learned_control":a['geometric_mean_joint']/best}
    return records,summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shared-module',required=True);ap.add_argument('--inventory',required=True);ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',required=True);a=ap.parse_args()
    m=load_shared(Path(a.shared_module));families=m.load_families(Path(a.inventory),Path(a.data_dir));records,summary=run(m,families);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/'development_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    with (out/'per_family.csv').open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    compact={"family_count":summary['family_count'],"development_gate":summary['development_gate'],"safe_atlas":summary['methods']['safe_atlas'],"fixed":summary['methods']['fixed'],"tangent":summary['tangent_mode_projective_correlation'],"atlas_diagnostics":summary['atlas_diagnostics']};print('V019_LOWRANK_FIVE_PROBE='+json.dumps(compact,sort_keys=True));return 0 if summary['development_gate']['pass'] else 2

if __name__=='__main__':raise SystemExit(main())
