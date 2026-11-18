from pathlib import Path
import zipfile, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT.parent/'Student folder.zip'
for d in ['src','app','ui','tests','configs','docs','artifacts','results/eda','results/models','results/tuning','results/hmm','results/maps','results/logs','data/train','data/student_test','data/evaluator_test','scripts']:
    (ROOT/d).mkdir(parents=True,exist_ok=True)
manifest=[]
with zipfile.ZipFile(source) as z:
    for info in z.infolist():
        parts=Path(info.filename).parts
        if info.is_dir() or len(parts)!=4 or parts[1] not in ('train','student_test'): continue
        target=ROOT/'data'/Path(*parts[1:])
        if target.suffix not in ('.track','.nodes','.arcs','.route'): continue
        target.parent.mkdir(parents=True,exist_ok=True)
        blob=z.read(info)
        if target.exists():
            assert target.read_bytes()==blob, f'Refusing overwrite: {target}'
        else: target.write_bytes(blob)
        manifest.append({'path':str(target.relative_to(ROOT)), 'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
p=ROOT/'results/eda/source_manifest.json'
if not p.exists(): p.write_text(json.dumps(manifest,indent=2))
print('Verified/extracted',len(manifest),'files; originals unchanged')