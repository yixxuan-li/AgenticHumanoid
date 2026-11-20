"""Continue the already-authorized reconstruction after the model transfer finishes."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--deadline-unix',type=float,required=True);args=p.parse_args()
checkpoint=Path('checkpoints/vggt_model_ranges.pt');out=Path('outputs/geometry');out.mkdir(parents=True,exist_ok=True)
status=out/'pipeline_status.json'
while not checkpoint.exists():
    status.write_text(json.dumps(dict(stage='waiting_for_model_download',updated_unix=time.time(),deadline_unix=args.deadline_unix),indent=2))
    if time.time()>args.deadline_unix:
        status.write_text(json.dumps(dict(stage='download_not_completed_before_deadline',updated_unix=time.time()),indent=2));sys.exit(2)
    time.sleep(20)
commands=[['scripts/run_vggt.py','--stride','2'],['scripts/process_geometry.py'],['scripts/render_photo.py']]
for command in commands:
    status.write_text(json.dumps(dict(stage='running',command=command,updated_unix=time.time()),indent=2))
    with (out/(Path(command[0]).stem+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status.write_text(json.dumps(dict(stage='failed',command=command,returncode=result.returncode,updated_unix=time.time()),indent=2));sys.exit(result.returncode)
status.write_text(json.dumps(dict(stage='ready_for_visual_and_geometric_review',updated_unix=time.time()),indent=2))
