"""Run the seven user-requested scene probes sequentially; stop on quota errors."""
import json
from pathlib import Path
import subprocess
import time

stage=Path('/opt/omnissiah/staging/scene-verification')
config=json.loads(Path('/opt/omnissiah/app/config.json').read_text())['profiles']['dell_query']['llm']
assert config.get('provider')=='gemini', 'Provider changed; obtain authorization before generation tests.'
cases=json.loads((stage/'scene_cases.json').read_text())
reports=[]
for case in cases:
    name=case['id']
    print('Testing narrative:',name,flush=True)
    output=stage/(name+'-narrate.json')
    log=stage/(name+'-narrate.log')
    started=time.monotonic()
    with log.open('w',encoding='utf-8') as destination:
        result=subprocess.run(['/opt/omnissiah/app/.venv/bin/python','-u',str(stage/'verify_scenes.py'),
                               '--mode','narrate','--case',name,'--output',str(output)],
                              stdout=destination,stderr=subprocess.STDOUT,timeout=360)
    if output.exists():
        row=json.loads(output.read_text())[0]
        reports.append(row)
        print(json.dumps({'case':name,'ok':row.get('ok'),'answered':row.get('answered'),
                          'characters':len(row.get('answer','')),'sources':[s.get('source') for s in row.get('sources',[])],
                          'seconds':round(time.monotonic()-started,2),
                          'tail':row.get('answer','')[-120:]}),flush=True)
    else:
        reports.append({'case':name,'ok':False,'error':'test did not produce a report'})
    content=log.read_text()
    if 'HTTP 429' in content or 'quota cooldown' in content:
        print('Quota exhausted; remaining generation probes deferred.',flush=True)
        break
(stage/'narrative-summary.json').write_text(json.dumps(reports,indent=2))
