"""Rebase stored metadata paths and dependent file checksums, preserving raw data."""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tarfile

from crop_pipeline.common import save_json
from crop_pipeline.storage_migration import LEGACY_ROOTS, DESTINATION

TEXT_SUFFIXES = {'.json', '.geojson', '.csv', '.md', '.txt', '.log', '.jsonl', '.yaml', '.yml'}
CHUNK = 4 * 1024 * 1024


def replacements(repository):
    root = Path(repository).resolve()
    result = {}
    for name in reversed(LEGACY_ROOTS):
        result[str(root / name).encode()] = str(root / DESTINATION).encode()
        result[('./'+name).encode()] = ('./'+DESTINATION).encode()
        result[name.encode()] = DESTINATION.encode()
    return result


def tokens(stream, paths, hashes=None, output=None):
    """Stream large tables; keep original transfer IDs and content-addressed names."""
    protected = re.compile(rb'"original_handoff_hashes"\s*:\s*\[[^\]]*\]')
    # A digest in a filename is an identity, not a file-checksum field.
    digest_pattern = re.compile(rb'(?<![/0-9a-zA-Z_])[0-9a-f]{64}(?![/0-9a-zA-Z_.])')
    references, path_change = set(), False
    old_digest, new_digest = hashlib.sha256(), hashlib.sha256()
    size = 0
    carry = b''

    def transform(piece):
        nonlocal path_change
        for old, new in sorted(paths.items(), key=lambda item: len(item[0]), reverse=True):
            if old in piece:
                path_change = True
                piece = piece.replace(old, new)
        def replace(match):
            value=match.group();references.add(value.decode('ascii'))
            return (hashes or {}).get(value,value)
        return digest_pattern.sub(replace,piece)

    while True:
        block=stream.read(CHUNK);old_digest.update(block)
        content=carry+block
        limit=max(0,len(content)-65536) if block else len(content)
        protected_matches=list(protected.finditer(content))
        # Only inspect boundary neighborhoods for split literal paths/digests.
        for old in paths:
            start=content.find(old,max(0,limit-len(old)))
            if start >= 0 and start < limit < start+len(old):limit=start
        for match in digest_pattern.finditer(content,max(0,limit-65),min(len(content),limit+65)):
            if match.start()<limit<match.end():limit=match.start()
        for match in protected_matches:
            if match.start()<limit<match.end():limit=match.start()
        pieces=[];start=0
        for match in protected_matches:
            if match.end()>limit:break
            pieces += [transform(content[start:match.start()]),match.group()]
            start=match.end()
        pieces.append(transform(content[start:limit]))
        changed=b''.join(pieces)
        if output is not None:output.write(changed)
        new_digest.update(changed);size+=len(changed)
        carry=content[limit:]
        if not block:break
    return old_digest.hexdigest(),new_digest.hexdigest(),references,path_change,size


def metadata_files(repository):
    """Walk authoritative files once, never follow output navigation links."""
    root = Path(repository).resolve()
    for parent in (root/'data/pipeline_work', root/'data/external', root/'pipeline_outputs'):
        for folder, directories, files in os.walk(parent, followlinks=False):
            directories[:] = [d for d in directories if not (Path(folder)/d).is_symlink()
                              and not (parent.name=='pipeline_outputs' and Path(folder)==parent and d=='transfers')]
            for name in sorted(files):
                path=Path(folder)/name
                if not path.is_symlink() and path.suffix in TEXT_SUFFIXES:
                    yield path
    settings=root/'settings.local.json'
    if settings.exists():
        yield settings


def repair_recorded_sizes(value, sizes):
    """Update file-byte metadata only when its corresponding checksum was rebased."""
    changed=False
    if isinstance(value,dict):
        if value.get('sha256') in sizes and isinstance(value.get('bytes'),int):
            size=sizes[value['sha256']]
            if value['bytes']!=size:value['bytes']=size;changed=True
        for item in value.values():changed=repair_recorded_sizes(item,sizes) or changed
    elif isinstance(value,list):
        for item in value:changed=repair_recorded_sizes(item,sizes) or changed
    return changed


def relocate(repository, *, resume_index=None):
    root=Path(repository).resolve();paths=replacements(root)
    for name in LEGACY_ROOTS:
        alias=root/name
        if not alias.is_symlink() or alias.resolve()!=root/DESTINATION:
            raise ValueError(f'Expected migration compatibility link: {alias}')
    if resume_index is not None:
        run=Path(resume_index).resolve()
        if (run/'migration.json').exists():
            raise ValueError('Only an interrupted, unmodified index can be resumed')
        db=sqlite3.connect(run/'dependencies.sqlite')
    else:
        run=root/'data/storage_migrations'/datetime.now(timezone.utc).strftime('rebase_%Y%m%dT%H%M%S%fZ')
        run.mkdir(parents=True)
        db=sqlite3.connect(run/'dependencies.sqlite')
        db.executescript('CREATE TABLE files(id INTEGER PRIMARY KEY,path TEXT,old_sha TEXT,size INTEGER,direct INTEGER);'
                         'CREATE TABLE refs(file_id INTEGER,digest TEXT,PRIMARY KEY(file_id,digest));'
                         'CREATE INDEX reverse_refs ON refs(digest);')
    done={r[0] for r in db.execute('SELECT path FROM files')}
    number=db.execute('SELECT COALESCE(MAX(id),0) FROM files').fetchone()[0]
    print(f'Indexing metadata paths and checksum dependencies ({number} already indexed)...',flush=True)
    for path in metadata_files(root):
        if str(path) in done:continue
        number+=1
        with path.open('rb') as stream:
            old,_,refs,direct,_=tokens(stream,paths)
        db.execute('INSERT INTO files VALUES(?,?,?,?,?)',(number,str(path),old,path.stat().st_size,int(direct)))
        db.executemany('INSERT INTO refs VALUES(?,?)',((number,d) for d in refs))
        if number%10000==0:
            db.commit();print(f'Indexed {number} metadata files',flush=True)
    db.commit()
    rows={r[0]:r for r in db.execute('SELECT * FROM files')}
    affected={i for i,r in rows.items() if r[4]}
    pending=deque(affected)
    while pending:
        i=pending.popleft()
        for (parent,) in db.execute('SELECT file_id FROM refs WHERE digest=?',(rows[i][2],)):
            if parent not in affected:
                affected.add(parent);pending.append(parent)
    by_hash=defaultdict(set)
    for i in affected:
        by_hash[rows[i][2]].add(i)
    deps={};parents=defaultdict(set)
    for i in affected:
        deps[i]=set()
        for (digest,) in db.execute('SELECT digest FROM refs WHERE file_id=?',(i,)):
            deps[i].update(by_hash.get(digest,()))
        for child in deps[i]:parents[child].add(i)
    queue=deque(i for i in affected if not deps[i]);order=[]
    while queue:
        i=queue.popleft();order.append(i)
        for parent in parents[i]:
            deps[parent].discard(i)
            if not deps[parent]:queue.append(parent)
    if len(order)!=len(affected):
        raise ValueError('Cyclic metadata checksum dependencies; nothing changed')
    # Save links outside the work tree too, including any old output navigation.
    links=[]
    for parent in (root/'pipeline_outputs',root/'data/pipeline_work'):
        for folder,dirs,files in os.walk(parent,followlinks=False):
            for name in dirs+files:
                path=Path(folder)/name
                if path.is_symlink():
                    target=path.resolve()
                    links.append({'path':str(path),'original_target':os.readlink(path),'target':str(target)})
            dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
    state={'status':'planned','files':len(rows),'affected':len(order),'links':links,
           'legacy_roots':list(LEGACY_ROOTS),'destination':DESTINATION}
    save_json(run/'migration.json',state)
    print(f'Backing up {len(order)} metadata files before editing...',flush=True)
    with tarfile.open(run/'original_metadata.tar.gz','w:gz',compresslevel=1) as archive:
        for i in order:
            p=Path(rows[i][1]);archive.add(p,arcname=p.relative_to(root))
    # Keep room for the largest single streaming replacement plus a margin.
    import shutil
    if shutil.disk_usage(root).free < max((rows[i][3] for i in order),default=0)+256*1024**2:
        raise OSError("Insufficient free space for an atomic metadata replacement; originals unchanged")
    changed_hashes={};new_sizes={};changes=[]
    try:
        for number,i in enumerate(order,1):
            _,name,old,_,_=rows[i];path=Path(name);temporary=path.with_name(path.name+'.relocating')
            if temporary.exists():raise FileExistsError(temporary)
            with path.open('rb') as source,temporary.open('xb') as output:
                actual,new,_,_,size=tokens(source,paths,changed_hashes,output)
            if actual!=old:
                temporary.unlink();raise ValueError(f'Metadata changed during migration: {path}')
            if path.suffix in {'.json','.geojson'} and size < 16*1024**2:
                value=json.loads(temporary.read_bytes())
                if repair_recorded_sizes(value,new_sizes):
                    temporary.write_text(json.dumps(value,indent=2)+'\n')
                    content=temporary.read_bytes();new=hashlib.sha256(content).hexdigest();size=len(content)
            new_sizes[new]=size
            if old.encode() in changed_hashes and changed_hashes[old.encode()]!=new.encode():
                raise ValueError('Identical original metadata rebased inconsistently')
            changed_hashes[old.encode()]=new.encode()
            temporary.chmod(path.stat().st_mode & 0o777)
            changes.append({'path':str(path),'original_sha256':old,'sha256':new,'bytes':size})
            temporary.replace(path)
            if number%5000==0:print(f'Rebased {number}/{len(order)} metadata files',flush=True)
        save_json(run/'changes.json',changes)
        # Normalize navigation links before removing the legacy entry points.
        for link in links:
            path=Path(link['path']);target=Path(link['target'])
            if path.resolve()!=target:raise ValueError(f'Output target changed: {path}')
            path.unlink();path.symlink_to(os.path.relpath(target,path.parent),target_is_directory=target.is_dir())
        state['status']='rebased_links_retained_for_validation'
        save_json(run/'migration.json',state)
    except BaseException:
        # Roll back only our writes. Never overwrite a concurrently edited file.
        from crop_pipeline.windows.seasons import file_sha256
        written={entry['path']:entry for entry in changes}
        conflicts=[]
        with tarfile.open(run/'original_metadata.tar.gz','r:gz') as archive:
            for member in archive:
                path=root/member.name
                entry=written.get(str(path))
                if entry is None:continue
                digest=file_sha256(path) if path.is_file() else None
                if digest==entry['sha256']:
                    archive.extract(member,root,filter='data')
                elif digest!=entry['original_sha256']:
                    conflicts.append(str(path))
        for link in links:
            path=Path(link['path'])
            expected=os.path.relpath(Path(link['target']),path.parent)
            if path.is_symlink() and os.readlink(path) in {expected,link['original_target']}:
                path.unlink();path.symlink_to(link['original_target'])
            else:conflicts.append(str(path))
        state['status']='rollback_conflicts' if conflicts else 'rolled_back'
        state['preserved_concurrent_edits']=conflicts
        save_json(run/'migration.json',state)
        raise
    finally:
        db.close()
    print(f'Metadata rebased; validate before unlinking aliases. Record: {run}',flush=True)
    return run


def finalize(run):
    """Verify rebased bytes before removing only the two known compatibility links."""
    run=Path(run).resolve();root=run.parents[2]
    state=json.loads((run/'migration.json').read_text())
    if state['status']!='rebased_links_retained_for_validation':
        raise ValueError('Expected a completed metadata rebase awaiting validation')
    changes=json.loads((run/'changes.json').read_text())
    print('Verifying rebased metadata before unlinking aliases...',flush=True)
    for change in changes:
        path=Path(change['path'])
        digest=hashlib.sha256();carry=b'';legacy=False
        with path.open('rb') as stream:
            while block:=stream.read(CHUNK):
                digest.update(block)
                legacy=legacy or b'sen4agrinet_work' in carry+block
                carry=block[-32:]
        if path.stat().st_size!=change['bytes'] or digest.hexdigest()!=change['sha256']:
            raise ValueError(f'Rebased metadata changed: {path}')
        if legacy:
            raise ValueError(f'Legacy storage path remains: {path}')
    for name in LEGACY_ROOTS:
        path=root/name
        if not path.is_symlink() or path.resolve()!=root/DESTINATION:
            raise ValueError(f'Expected original compatibility link: {path}')
    for name in LEGACY_ROOTS:
        (root/name).unlink()
    state['status']='verified_aliases_removed'
    save_json(run/'migration.json',state)
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path.cwd())
    parser.add_argument('--resume-index',type=Path,help='Resume an interrupted metadata index before any edits')
    parser.add_argument('--finalize',type=Path,help='Verify a completed rebase and remove its compatibility links')
    args=parser.parse_args()
    if args.finalize:
        finalize(args.finalize)
    else:
        relocate(args.repository,resume_index=args.resume_index)
