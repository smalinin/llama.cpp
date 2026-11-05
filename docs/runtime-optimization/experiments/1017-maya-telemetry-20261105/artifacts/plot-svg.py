from pathlib import Path
import sqlite3,sys,json,html,math
from importlib.machinery import SourceFileLoader
m=SourceFileLoader('timeline',str(Path(__file__).parent/'analyze.py')).load_module()
p=Path(sys.argv[1]);bin_ms=25;step=25_000_000;W=1200;left=100;right=25;scale=(W-left-right)/6500;row=24;panels=[];summary={}
for engine in ['maya','ours']:
 c=sqlite3.connect(p/engine/'capture.1.sqlite');c.text_factory=lambda x:x.decode('utf8','replace');s=dict(c.execute('select id,value from StringIds'));lo,hi=next((a,b)for a,b,n,i in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')if(n or s.get(i,''))=='phase.prefill');prod={d:[]for d in range(6)};comm={d:[]for d in range(6)}
 for a,b,d,i in c.execute('select start,end,deviceId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?',(lo,hi)):
  (comm if m.family(s[i])=='communication_and_wait'else prod)[d].append((max(a,lo),min(b,hi)))
 prod={d:m.merge(v)for d,v in prod.items()};comm={d:m.merge(v)for d,v in comm.items()};panels.append((engine,lo,hi,prod,comm))
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="505" viewBox="0 0 {W} 505">','<rect width="100%" height="100%" fill="white"/>','<g font-family="sans-serif" font-size="12" fill="#243340">','<text x="20" y="25" font-size="18">Prefill 8192: recorded kernel coverage per GPU</text>','<text x="20" y="46">25 ms bins; blue = compute/other kernels; amber strip = transport/wait kernels. Hover for coverage.</text>','<text x="20" y="65">Diagnostic capture, not native speed or SM occupancy. DMA is not plotted; warnings are documented in REPORT.md.</text>']
for pi,(engine,lo,hi,prod,comm)in enumerate(panels):
 y0=100+pi*190;svg.append(f'<text x="20" y="{y0-10}" font-size="15">{engine}</text>');endms=(hi-lo)/1e6;svg.append(f'<text x="{left+endms*scale+5:.1f}" y="{y0+71}" font-size="11">{endms:.0f} ms</text>')
 for d in range(6):
  y=y0+d*row;svg.append(f'<text x="30" y="{y+14}">GPU {d}</text>');svg.append(f'<rect x="{left}" y="{y}" width="{endms*scale:.2f}" height="19" fill="#f2f5f8"/>')
  for start in range(lo,hi,step):
   end=min(start+step,hi);cov=lambda ints:sum(max(0,min(b,end)-max(a,start))for a,b in ints if b>start and a<end)/(end-start)
   pc,cc=cov(prod[d]),cov(comm[d]);x=left+(start-lo)/1e6*scale;w=(end-start)/1e6*scale+.02
   if pc:svg.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="15" fill="#157cba" fill-opacity="{pc:.3f}"><title>{engine} GPU{d}, {(start-lo)/1e6:.0f} ms: compute coverage {100*pc:.1f}%</title></rect>')
   if cc:svg.append(f'<rect x="{x:.2f}" y="{y+16}" width="{w:.2f}" height="3" fill="#d99125" fill-opacity="{cc:.3f}"><title>transport/wait coverage {100*cc:.1f}% (may overlap compute)</title></rect>')
 for t in range(0,6501,1000):
  x=left+t*scale;svg.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+144}" stroke="#ccd5dc" stroke-width=".5"/><text x="{x:.1f}" y="{y0+161}" text-anchor="middle">{t}</text>')
svg.extend(['<text x="600" y="493" text-anchor="middle">Time from phase start (ms)</text>','</g></svg>']);(p/'prefill8k-timeline.svg').write_text('\n'.join(svg));print('SVG saved')
