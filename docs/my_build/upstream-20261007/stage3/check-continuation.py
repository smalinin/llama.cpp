"""Check a continuation against a saved full prefix without recurrent rewind."""
import json

def check_continuation(api,port,directory,result,prompt,allow_full_prefill=False,prior_response='state-old-version-response.json'):
    def save(label,data):
        (directory/(label+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    def slot(label,action,filename=None):
        request={} if filename is None else {'filename':filename}
        response=api(port,'/slots/0?action='+action,request)
        save(label+'-request',request);save(label+'-response',response)
        if 'error' in response:raise RuntimeError('continuation slot action failed: '+str(response))
        return response
    last=json.loads((directory/prior_response).read_text())
    token_request={'content':prompt,'add_special':True,'parse_special':True}
    token_response=api(port,'/tokenize',token_request)
    save('continuation-tokenize-request',token_request);save('continuation-tokenize-response',token_response)
    prefix=token_response['tokens']+last['tokens']
    saved=slot('continuation-save','save','continuation.bin')
    request={'prompt':prefix,'n_predict':32,'temperature':0,'seed':1234,'top_k':40,'top_p':.95,'min_p':.05,
             'cache_prompt':True,'return_tokens':True,'stream':False}
    save('continuation-request',request)
    control=api(port,'/completion',request);save('continuation-control-response',control)
    slot('continuation-erase','erase')
    slot('continuation-restore','restore','continuation.bin')
    restored=api(port,'/completion',request);save('continuation-restored-response',restored)
    expected=len(prefix)-1
    report={'prefix_tokens':len(prefix),'saved_tokens':saved.get('n_saved'),
            'expected_cached_tokens':expected,'control_timings':control.get('timings'),
            'restored_timings':restored.get('timings'),'tokens_equal':control.get('tokens')==restored.get('tokens'),
            'text_equal':control.get('content')==restored.get('content')}
    result['continuation']=report;save('continuation-result',report)
    if 'error' in control or 'error' in restored:raise RuntimeError('continuation completion failed')
    cached=all(r.get('timings',{}).get('cache_n')==expected for r in [control,restored])
    report['both_requests_used_saved_prefix']=cached
    report['scope']='target slot file; draft/speculative state is not serialized by the existing server'
    save('continuation-result',report)
    if not cached:
        if allow_full_prefill:
            report['limitation']='speculative continuation falls back to full prefill; native target continuation is checked separately'
            save('continuation-result',report)
            return
        raise RuntimeError('continuation did not reuse the saved prefix')
    if not report['tokens_equal'] or not report['text_equal']:raise RuntimeError('continuation differs after restore')
