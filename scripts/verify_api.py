import os
import json
from typing import Dict, List
from kimina_client import KiminaSandboxClient, Snippet, KiminaClient

def format_results(results):

    returns = []

    responses = [r.response for r in results]

    for r in responses:

        format_res = {
            'info':{
                'system_errors': None, 
                'sorries': [],
                'errors': [], 
                'warnings': []
                }
            }
        
        format_res['info']['sorries'] = r['sorries'] if 'sorries' in r else []

        for item in r['messages']:
            if item['severity'] == 'warning':
                format_res['info']['warnings'].append(item)
            if item['severity'] == 'error':
                format_res['info']['errors'].append(item)
            if item['severity'] == 'system_errors':
               format_res['info']['system_errors'] = item

        format_res['pass'] = len(format_res['info']['errors']) == 0

        returns.append(format_res)

    return returns

def batch_verify_lean_codes(lean_contents: List[str], concurrency: int = 5, server_timeout=600) -> List[Dict]:
    
    client = KiminaClient(api_url=f"http://localhost:8000", http_timeout=self.server_timeout)
    
    snips = [Snippet(id=f"{idx}", code=str(proof)) for idx, proof in enumerate(lean_contents)]
    
    response = client.check(snips=snips, max_workers=concurrency)
    
    results = format_results(sorted(response.results, key=lambda result: int(result.id)))
    
    return results
