"""Content identity for a simulator scene and its visual assets."""
import hashlib
from pathlib import Path


def scene_fingerprint(directory):
    directory=Path(directory)
    files=[directory/p for p in ['scene.xml','scene.usda','manifest.json']]
    files+=sorted((directory/'textures').glob('*.png'))
    files+=sorted((directory/'photo').glob('*.png'))
    files+=sorted((directory/'photo').glob('*.obj'))
    files+=sorted((directory/'visual').glob('*.png'))
    files+=sorted((directory/'visual').glob('*.obj'))
    hashes={}
    for path in files:
        with path.open('rb') as f:hashes[str(path.relative_to(directory))]=hashlib.file_digest(f,'sha256').hexdigest()
    digest=hashlib.sha256()
    for name,value in sorted(hashes.items()):digest.update((name+'\0'+value+'\n').encode())
    return dict(sha256=digest.hexdigest(),files=hashes)
