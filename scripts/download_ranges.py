"""Resumable byte-range download with bounded concurrency and verified ranges."""
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
import requests
import hashlib
import shutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('url');p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--chunk-mb',type=int,default=16)
    p.add_argument('--sha256',help='Expected complete-file SHA-256')
    p.add_argument('--keep-parts',action='store_true')
    args=p.parse_args()
    def checksum(path):
        with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
    if args.sha256 and args.output.exists() and checksum(args.output)==args.sha256:
        print(json.dumps(dict(path=str(args.output),bytes=args.output.stat().st_size,status='already verified',sha256=args.sha256,checksum_verified=True)),flush=True);return
    response=requests.head(args.url,allow_redirects=True,timeout=60)
    response.raise_for_status()
    size=int(response.headers['Content-Length'])
    if args.output.exists() and args.output.stat().st_size==size:
        digest=checksum(args.output) if args.sha256 else None
        if args.sha256 and digest!=args.sha256:raise RuntimeError('Existing complete-length file has the wrong checksum; inspect or choose a fresh output')
        print(json.dumps(dict(path=str(args.output),bytes=size,status='already present',sha256=digest,checksum_verified=bool(args.sha256))),flush=True);return
    chunks=args.output.with_suffix(args.output.suffix+'.parts');chunks.mkdir(parents=True,exist_ok=True)
    step=args.chunk_mb*1024**2
    tasks=[(i,min(i+step,size)-1) for i in range(0,size,step)]
    def fetch(pair):
        a,b=pair;path=chunks/f'{a:012d}'
        if path.exists() and path.stat().st_size==b-a+1:return
        for attempt in range(5):
            try:
                # Resolve each request afresh so signed redirects can expire during long downloads.
                r=requests.get(args.url,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(40,120))
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'):
                    raise RuntimeError(f'Server ignored range: {r.status_code} {r.headers.get("Content-Range")}')
                tmp=path.with_suffix('.partial')
                with tmp.open('wb') as out:
                    for block in r.iter_content(1024*256):out.write(block)
                if tmp.stat().st_size!=b-a+1:raise RuntimeError('Truncated range')
                tmp.replace(path);print(f'{a}-{b} complete',flush=True);return
            except Exception as exc:
                print(f'{a} retry {attempt}: {exc}',flush=True);time.sleep(min(2**attempt,15))
        raise RuntimeError(f'Failed range {a}-{b}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch,tasks))
    tmp=args.output.with_suffix(args.output.suffix+'.partial')
    with tmp.open('wb') as out:
        for a,b in tasks:
            with (chunks/f'{a:012d}').open('rb') as inp:
                while block:=inp.read(1024**2):out.write(block)
    if tmp.stat().st_size!=size:raise RuntimeError('Final length mismatch')
    digest=checksum(tmp) if args.sha256 else None
    if args.sha256 and digest!=args.sha256:raise RuntimeError(f'Assembled checksum mismatch: {digest}')
    tmp.replace(args.output)
    if not args.keep_parts:shutil.rmtree(chunks)
    print(json.dumps(dict(path=str(args.output),bytes=size,sha256=digest,checksum_verified=bool(args.sha256))),flush=True)


if __name__=='__main__':main()
