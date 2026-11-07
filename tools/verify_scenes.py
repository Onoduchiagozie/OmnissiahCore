"""Scene verification CLI. Default mode inspects retrieval without LLM calls."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import requests


def cases_path():
    return Path(__file__).with_name('scene_cases.json')


def has_scene_answer(text, sources):
    normalized=' '.join(text.split()).strip().lower()
    refusals=(
        'the archives do not hold enough of this moment',
        'the archives do not clearly record enough here',
        'the chronicles do not record this',
        'the chronicles do not clearly record enough here',
    )
    return bool(sources) and bool(normalized) and '[error]' not in normalized and not normalized.startswith(refusals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--mode', choices=('inspect','focus','explore','narrate','focus/stream'), default='inspect')
    parser.add_argument('--case', default='all', help='case id, all, or list')
    wording = parser.add_mutually_exclusive_group()
    wording.add_argument('--raw', dest='raw', action='store_true', default=True, help='send original human wording (default)')
    wording.add_argument('--expanded', dest='raw', action='store_false', help='use the explicit diagnostic query for comparison')
    parser.add_argument('--book-filter', help='optional explicit source substring')
    parser.add_argument('--cases', type=Path, default=cases_path())
    parser.add_argument('--output', type=Path, help='save a JSON report')
    args=parser.parse_args()
    cases=json.loads(args.cases.read_text(encoding='utf-8'))
    if args.case=='list':
        for case in cases: print(case['id']+': '+case['raw_query'])
        return
    selected=cases if args.case=='all' else [case for case in cases if case['id']==args.case]
    if not selected: parser.error('unknown case id; use --case list')
    if args.mode!='inspect' and args.case=='all':
        parser.error('choose one --case for generation to avoid consuming quota with a batch')
    rows=[]
    for case in selected:
        query=case['raw_query'] if args.raw else case['query']
        payload={'query':query,'session_id':'scene-test-'+uuid.uuid4().hex}
        if args.book_filter: payload['book_filter']=args.book_filter
        started=time.monotonic()
        print('\nCASE: '+case['id']+'\nQUERY: '+query,flush=True)
        try:
            streaming=args.mode in ('narrate','focus/stream')
            with requests.post(args.base_url.rstrip('/')+'/query/'+args.mode,json=payload,
                               headers={'ngrok-skip-browser-warning':'1'},stream=streaming,timeout=(10,300)) as response:
                response.raise_for_status()
                if streaming:
                    text=[]; sources=[]; stages=[]; events=[]; current=[]; done=0; error=False
                    first_text_seconds=None
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        if line is None: continue
                        if line.startswith('data:'):
                            field=line[5:]
                            current.append(field[1:] if field.startswith(' ') else field)
                        elif not line and current:
                            event='\n'.join(current); current=[]; events.append(event)
                            if event=='[DONE]': done+=1
                            elif event.startswith('__SOURCES__:'): sources=json.loads(event.split(':',1)[1])
                            elif event.startswith('__STAGE__:'): stages.append(event.split(':',1)[1])
                            elif event.startswith('[ERROR]'):
                                error=True; print(event,flush=True)
                            else:
                                if first_text_seconds is None: first_text_seconds=round(time.monotonic()-started,2)
                                text.append(event); print(event,end='',flush=True)
                    print()
                    answer=''.join(text)
                    row={'case':case['id'],'query':query,'ok':not error and done==1 and 'complete' in stages,
                         'answered':not error and has_scene_answer(answer,sources),
                         'requires_source_review':not error and has_scene_answer(answer,sources),
                         'answer':answer,'sources':sources,'stages':stages,'done_events':done,
                         'first_text_seconds':first_text_seconds}
                else:
                    data=response.json()
                    if args.mode=='inspect':
                        inspection=data['inspection']
                        hits=inspection.get('stitched_hits',[])
                        row={'case':case['id'],'query':query,'ok':True,'kind':'raw retrieval only, not answer-quality proof',
                             'query_plan':inspection.get('query_plan'),'hits':len(hits),
                             'sources':[{'source':hit.get('source'),'range':hit.get('stitch_range'),
                                         'overlap':hit.get('query_overlap_terms'),
                                         'preview':hit.get('text','')[:500]} for hit in hits]}
                    else:
                        row={'case':case['id'],'query':query,'ok':bool(data.get('response')) and '[ERROR]' not in data.get('response',''),
                             'answered':has_scene_answer(data.get('response',''),data.get('sources',[])),
                             'requires_source_review':has_scene_answer(data.get('response',''),data.get('sources',[])),**data}
                    print(json.dumps(row,indent=2),flush=True)
            row['seconds']=round(time.monotonic()-started,2)
        except (requests.RequestException, ValueError, KeyError) as exc:
            row={'case':case['id'],'query':query,'ok':False,'error':str(exc)}
            print(json.dumps(row),flush=True)
        rows.append(row)
    if args.output: args.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if any(not row['ok'] for row in rows): sys.exit(1)


if __name__=='__main__': main()
