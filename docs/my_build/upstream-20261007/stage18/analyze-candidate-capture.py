from pathlib import Path
import hashlib,json,runpy
R=Path(__file__).resolve().parent
s=(R/'analyze-capture.py').read_text()
s=s.replace("D=R/'capture-output'","D=R/'candidate-capture-output'")
s=s.replace("ref=R.parent/'stage17/explain-replay-output'/p.name", "ref=R/'candidate-output'/p.name")
s=s.replace("R/'capture-summary.json'","R/'candidate-capture-summary.json'")
exec(compile(s,str(R/'analyze-capture.py'),'exec'))
