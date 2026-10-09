from pathlib import Path
import json,struct
r=Path(__file__).resolve().parent
for dst,src,key in [('long-prompt.bin','before-off/ratio2-long-fresh-request.json','prompt'),('long-tokens.bin','before-off/ratio2-long-fresh-response.json','tokens')]:
 v=json.loads((r/src).read_text())[key];(r/dst).write_bytes(struct.pack('<'+'i'*len(v),*v))
