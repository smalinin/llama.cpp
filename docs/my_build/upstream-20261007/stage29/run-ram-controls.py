#!/usr/bin/env python3
"""Bounded controls for the observed RAM-restore differences."""
from pathlib import Path
import importlib.util,json,resource
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ram_stage29',R/'run-ram.py');ram=importlib.util.module_from_spec(spec);spec.loader.exec_module(ram)
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 for label,kind in [('dflash-ram-off','draft-dflash'),('native-ram-off','none')]:
  result=ram.base.run(label,kind,0)
  p=R/'server-runs'/label/'control-manifest.json';meta=json.loads(p.read_text());meta['request_schedule']='fully sequential, automatic slots; three long distinct prompts; fresh1/reuse1/reuse2/fresh2';meta['driver_sources'].update({str(p):ram.base.sha(p) for p in [R/'run-ram.py',Path(__file__)]});ram.base.save(p,meta)
  if result['status']!='passed':raise SystemExit(1)
