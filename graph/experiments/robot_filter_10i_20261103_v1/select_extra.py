"""Predeclared new-parent selection using availability/identity, never outcomes."""
import json,hashlib,os
from pathlib import Path
here=Path(__file__).resolve().parent
frozen=Path(os.environ['TASK10I_ROOT'])
read=lambda p:json.loads(p.read_text())
old=read(here/'regression.json')['cases'];excluded={x['parent'] for x in old}
scenes=read(frozen/'protocol.json')['scenes'];eligible=[]
for s in scenes:
    if s['parent'] in excluded:continue
    def path(size,camera):
        name=s['parent']+'_track_t000' if s['cohort']=='tracking' else s['id']
        kind='tracking' if s['cohort']=='tracking' else 'static'
        return frozen/'masks'/'actual'/kind/f'r0__{name}__{size}__{camera}'/'masks.npz'
    if all(path(size,c).exists() and (frozen/'sensors'/s['id']/str(size)/(c+'.npz')).exists() for size in (128,512,768,1024) for c in ('base_camera','hand_camera')):eligible.append(s)
parents=sorted({s['parent'] for s in eligible},key=lambda p:hashlib.sha256(('task10i-roi-extra:'+p).encode()).hexdigest())[:6]
selected=[min((s for s in eligible if s['parent']==p),key=lambda s:hashlib.sha256(s['id'].encode()).hexdigest()) for p in parents]
cases=[dict(scene=s['id'],parent=s['parent'],cohort=s['cohort'],timestep=s['timestep'],stratum='extra_identity_selected',size=size) for s in selected for size in (128,512,768,1024)]
(here/'extra.json').write_text(json.dumps(dict(cases=cases,eligible_parent_count=len({s['parent'] for s in eligible}),rule=read(here/'protocol.json')['selection']),indent=2)+'\n')
print(json.dumps(dict(extra_parents=parents,conditions=len(cases))))
