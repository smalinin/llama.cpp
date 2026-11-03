from pathlib import Path
import json
P=Path(__file__).parent;O=Path('/tmp/dsv41-candidate-block-local-20261103')
s=(O/'bench.py').read_text().replace(";assert json.load(open(P/'operator-results.json'))['complete']",'').replace('previous pool=1 in all arms','actual pooling defaultOFF/unset in all arms; block OFF/default/OFF').replace('19077','19087').replace('candidate-block-screen','pool0-default-screen')
s=s.replace("[('off-pre','0'),('on','1'),('off-post','0')]", "[('off-pre','0'),('on',None),('off-post','0')]")
s=s.replace("e=base_env.copy();e['LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL']=value;assert'LD_PRELOAD'not in e", "e=base_env.copy();e.pop('LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL',None)\n if value is not None:e['LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL']=value\n assert 'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in e and 'LD_PRELOAD'not in e")
(P/'bench.py').write_text(s)
