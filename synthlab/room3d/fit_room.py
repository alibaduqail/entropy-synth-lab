"""Fit the upright room photos. SciPy is installed in ../thermal-datadriven/.venv/bin/python.
System python3 uses a deterministic NumPy damped Gauss-Newton solver for the same soft-L1 loss.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INITIAL_CAMERAS = [[.2,1.6,1.65,0,-22,0], [.2,1.6,1.65,0,-22,0],
                   [2.1,.1,1.8,105,-24,0], [2.4,1,1.75,165,-22,0],
                   [2.15,.65,1.35,110,-29,0]]
INITIAL = np.r_[ [2.65,3.1,2.55,1.15,1.3,.73,1.48,.16,.02,.68,1.52,1.45,1.25,
                  .75,1.4,1.4,1.6,1.02,545,850], np.ravel(INITIAL_CAMERAS),
                 [1.10,1.7,2.134,.914,1.2,2.15,1.4,.44]]
# (parameter index, prior mean, prior sigma), metres except the two focal lengths in pixels.
SHARED = [(18,544,45),(19,1090,130),(7,.045,.06),(8,.045,.06)]
METRIC = [(52,2.134,.04),(53,.914,.05),(54,1.2,.13),(55,2.15,.18),(56,1.4,.18),
          (57,.44,.025),(17,.953,.15),(6,1.50,.25),(5,.65,.12)]


def points(q):
    X,Y,H,x,y,w,l,hingey,dx,sy,rx,ry,rz,sty,thy,tx,tz,tw = q[:18]
    kx,ky,dz,dw,swH,stH,thH,klen = q[50:]
    h = .74
    return dict(kN=(kx,ky+klen/2,h+.02), kS=(kx,ky-klen/2,h+.02),
                neceil=(X,Y,H), nwceil=(0,Y,H), nefloor=(X,Y,0), nwfloor=(0,Y,0),
                swfloor=(0,0,0), nwdesk=(x,y+l,h), swdesk=(x,y,h),
                nedesk=(x+w,y+l,h), sedesk=(x+w,y,h), dtop=(X-dx,hingey+dw,dz),
                dhinge=(X-dx,hingey,dz), dbot=(X-dx,hingey+dw,0), dfoot=(X-dx,hingey,0),
                switch=(X,sy,swH), rig=(rx,ry,rz), stro=(0,sty,stH), ther=(0,thy,thH),
                tvl=(tx-tw/2,Y-.05,tz+tw*9/32), tvr=(tx+tw/2,Y-.05,tz+tw*9/32),
                tvb=(tx+tw/2,Y-.05,tz-tw*9/32))


def axes(pose):
    yaw,pitch,roll = np.radians(pose[3:6])
    forward = np.array([np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch), np.sin(pitch)])
    right = np.array([np.sin(yaw),-np.cos(yaw),0.])
    up = np.cross(right, forward)
    return right*np.cos(roll)+up*np.sin(roll), up*np.cos(roll)-right*np.sin(roll), forward


def project(xyz, pose, focal, wh):
    right,up,forward = axes(pose)
    v = np.asarray(xyz) - pose[:3]
    return wh/2 + focal * np.column_stack((v@right, -v@up)) / (v@forward)[:,None]


def solve(fun, start):
    try:
        from scipy.optimize import least_squares
    except ImportError:
        pass
    else:
        result = least_squares(fun,start,x_scale='jac',loss='soft_l1',max_nfev=300)
        if not result.success:
            raise RuntimeError(result.message)
        return result.x, 'scipy'
    q = np.array(start, dtype=float)
    damping = .001
    for _ in range(300):
        r = fun(q)
        step = 1e-6 * np.maximum(abs(q),1.)
        jac = np.column_stack([(fun(q+d)-r)/h for d,h in zip(np.diag(step), step)])
        weight = (1+r*r)**(-.25)
        weighted = weight[:,None]*jac
        normal = weighted.T @ weighted
        delta = np.linalg.solve(normal + damping*np.diag(np.maximum(normal.diagonal(),1e-12)),
                                -weighted.T @ (weight*r))
        improvement = np.sum(np.hypot(1,r)-np.hypot(1,fun(q+delta)))
        if improvement > 0:
            q += delta
            if improvement < 1e-9:
                return q, 'numpy'
            damping = max(1e-10,damping/3)
        else:
            damping *= 5
    raise RuntimeError('NumPy fit did not converge')


def camera(name, pose, focal, wh, err, keys):
    _,up,forward = axes(pose)
    hfov,vfov = np.degrees(2*np.arctan(wh/(2*focal)))
    return dict(id=name, position=pose[:3].tolist(), look_at=(pose[:3]+forward).tolist(),
                up=up.tolist(), image_wh=wh.tolist(), focal_px=float(focal),
                hfov_deg=float(hfov), vfov_deg=float(vfov), yaw_pitch_roll_deg=pose[3:6].tolist(),
                rms_px=float(np.sqrt(np.mean(np.sum(err**2,axis=1)))), n_landmarks=len(err),
                residual_xy_px=dict(zip(keys,err.tolist())))


def export_points(scene):
    boxes = {o['id']:o for o in scene['objects']}
    def spot(name, offset=(0,0,0)):
        obj = boxes[name]
        return np.array(obj['center'])+np.array(obj['size'])*offset/2

    X,Y,H = scene['room']['size_xyz']
    xyz = dict(neceil=(X,Y,H), nwceil=(0,Y,H), nefloor=(X,Y,0), nwfloor=(0,Y,0),
               swfloor=(0,0,0), switch=spot('light_switch'), rig=spot('mid360_body',(0,0,1)),
               stro=spot('fire_strobe'), ther=spot('thermostat'))
    for name,sx,sy in [('nwdesk',-1,1),('swdesk',-1,-1),('nedesk',1,1),('sedesk',1,-1)]:
        xyz[name] = spot('desk_top',(sx,sy,1))
    for name,sy,sz in [('dtop',1,1),('dhinge',-1,1),('dbot',1,-1),('dfoot',-1,-1)]:
        xyz[name] = spot('door',(-1,sy,sz))
    for name,sx,sz in [('tvl',-1,1),('tvr',1,1),('tvb',1,-1)]:
        xyz[name] = spot('tv',(sx,-1,sz))
    for name,sy in [('kN',1),('kS',-1)]:
        xyz[name] = spot('keyboard_primary',(0,sy,1))
    return xyz


def fit(data, scene):
    views = []
    for i in range(len(INITIAL_CAMERAS)):
        name = f'photo_{i+1}'
        photo = data['photos'][name]
        picks = photo['landmarks']
        sigma = [22 if i==4 else 24 if (i<2 and k in ('dbot','sedesk')) or
                 (i==3 and k=='nwfloor') else 7 for k in picks]
        views.append((name,list(picks),np.array(list(picks.values())),np.array(photo['image_wh']),
                      np.array(sigma)[:,None]))

    def errors(q):
        xyz = points(q)
        for i,(_,keys,pixels,wh,_) in enumerate(views):
            pose = q[20+6*i:26+6*i]
            yield project([xyz[k] for k in keys],pose,q[18+(i==4)],wh)-pixels

    def objective(q, priors):
        prior = [(q[j]-mean)/sigma for j,mean,sigma in priors]
        return np.concatenate([*( (e/v[-1]).ravel() for e,v in zip(errors(q),views)),prior])

    q,solver = solve(lambda p: objective(p,SHARED+METRIC),INITIAL)
    free,_ = solve(lambda p: objective(p,SHARED),INITIAL)
    references = {}
    for name,nominal,estimated in [('desk',.74,.74),('door_width',.914,free[53]),
                                   ('short_leaf',2.134,free[52]),('tall_leaf',2.438,free[52]),
                                   ('keyboard',.44,free[57]),('tv',.952,free[17]),
                                   ('switch',1.20,free[54]),('thermostat',1.4,free[56]),
                                   ('strobe',2.15,free[55])]:
        references[name] = dict(nominal_m=nominal,desk_scaled_m=float(estimated),
                                room_xyz_m=(free[:3]*nominal/estimated).tolist())
    comparisons = {}
    for leaf in ('short_leaf','tall_leaf'):
        factor = references[leaf]['nominal_m']/free[52]
        comparisons[leaf] = {}
        for ref in ('desk','keyboard','door_width','switch'):
            entry = references[ref]
            comparisons[leaf][ref] = 100*(factor*entry['desk_scaled_m']/entry['nominal_m']-1)

    cams = []
    for i,((name,keys,_,wh,_),err) in enumerate(zip(views,errors(q))):
        cams.append(camera(name,q[20+6*i:26+6*i],q[18+(i==4)],wh,err,keys))
    exported = export_points(scene)
    for i,(cam,(_,keys,pixels,wh,_)) in enumerate(zip(cams[:-1],views)):
        err = project([exported[k] for k in keys],q[20+6*i:26+6*i],q[18],wh)-pixels
        cam['export_rms_px'] = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
        cam['export_residual_xy_px'] = dict(zip(keys,err.tolist()))
    wh = np.array(data['photos']['photo_5']['image_wh'])
    original = cams[-1]
    picks = data['photo_5_refit']['landmarks']
    boxes = {o['id']:o for o in scene['objects']}
    coords, pixels, scales = [], [], []
    for p in picks.values():
        obj = boxes[p['box']]
        coords.append(obj['center']+np.array(obj['size'])*p['half_size_offset']/2)
        pixels.append(p['xy'])
        scales.append(p['sigma_px'])
    def new_errors(p):
        return project(coords,p[:6],p[6],wh)-pixels
    def objective5(p):
        return np.r_[(new_errors(p)/np.array(scales)[:,None]).ravel(),(p[6]-1090)/130]
    p,_ = solve(objective5,np.r_[q[44:50],q[19]])
    before = new_errors(np.r_[q[44:50],q[19]])
    cams[-1] = camera('photo_5',p[:6],p[6],wh,new_errors(p),picks)
    cams[-1]['export_rms_px'] = cams[-1]['rms_px']
    return dict(fitted_room_xyz_m=q[:3].tolist(), exported_room_xyz_m=scene['room']['size_xyz'],
                fitted_leaf_m=q[52], references=references, door_hypotheses_error_percent=comparisons,
                cameras=cams, original_photo_5=original,
                photo_5_before_refit_rms_px=float(np.sqrt(np.mean(np.sum(before**2,axis=1)))),
                settings=dict(solver=solver,loss='soft_l1',max_iterations=300,seed=None,
                              initialization=INITIAL.tolist(),shared_priors=SHARED,metric_priors=METRIC,
                              desk_height_m=.74, principal_point='image midpoint', rms='Euclidean pixel distance per fitted point, full resolution',
                              sigma_px_by_photo=[dict(zip(v[1],v[-1].ravel().tolist())) for v in views],
                              photo_5_focal_prior_px=[1090,130]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--landmarks',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scene',type=Path,default=ROOT/'scene/room_v1.json')
    args = parser.parse_args()
    report = fit(json.loads(args.landmarks.read_text()),json.loads(args.scene.read_text()))
    # Keep pixel pairs, parameter triples and camera vectors together on one line.
    import re
    formatted = json.dumps(report,indent=2,allow_nan=False)
    formatted = re.sub(r'\[\s*[-+0-9.eE]+(?:,\s*[-+0-9.eE]+)*\s*\]',
                       lambda m: json.dumps(json.loads(m.group())), formatted)
    args.out.write_text(formatted+'\n')


if __name__ == '__main__':
    main()
