import argparse
import csv
import gzip
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def sha256(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def deep_check():
    campaign=ROOT/'code/campaign'
    setup=json.loads((campaign/'results_fixed4/setup.json').read_text())
    workbook=ROOT/'code/data/slicing_dataset_pre_rl.xlsx'
    if sha256(workbook)!=setup['preprocessing']['input_sha256']:
        raise ValueError('Workbook does not match the archived input.')
    with zipfile.ZipFile(workbook) as wb:
        bad=wb.testzip()
        if bad is not None:
            raise ValueError(f'Workbook member failed CRC: {bad}')
    logs=sorted(ROOT.glob('*/campaign/results_*/logs/*.gz'))+sorted(ROOT.glob('reconstructed/*/*.gz'))
    expected_rows=len(setup['eval_starts'])*setup['horizon']*len(setup['preprocessing']['slice_order'])
    for log in logs:
        with gzip.open(log,'rb') as f:
            header=f.readline().decode().strip().split(',')
            if not {'slice','episode','t','miss','elig','I','L'}.issubset(header):
                raise ValueError(f'Unexpected columns: {log}')
            count=sum(block.count(b'\n') for block in iter(lambda:f.read(1024*1024), b''))
        if count!=expected_rows:
            raise ValueError(f'{log.name} has {count} records; expected {expected_rows}')
    checkpoints=sorted(ROOT.glob('*/campaign/results_fixed4/agents/*.pkl'))
    for ck in checkpoints:

        with ck.open('rb') as f:
            last=None
            for op, arg, pos in pickletools.genops(f):
                last=op
            if last is None or last.name!='STOP' or f.read(1):
                raise ValueError(f'Invalid checkpoint framing: {ck}')
    print(f'PASS: {len(logs)} gzip streams, each with {expected_rows:,} data rows; {len(checkpoints)} complete checkpoint streams; source workbook checksum and CRC.')
def main():
    p=argparse.ArgumentParser(description='Check the file hashes and, with --deep, the log row counts and checkpoint streams.');p.add_argument('--deep',action='store_true');args=p.parse_args()
    manifest=ROOT/'MANIFEST.sha256'
    if manifest.exists():
        lines=manifest.read_text().splitlines()
        for line in lines:
            digest,name=line.split('  ',1)
            path=ROOT/name
            if sha256(path)!=digest:raise ValueError(name+' checksum mismatch')
        print(f'PASS: {len(lines)} file checksums.')
    elif not args.deep:
        raise FileNotFoundError(manifest)
    if args.deep:deep_check()
if __name__=='__main__':main()
