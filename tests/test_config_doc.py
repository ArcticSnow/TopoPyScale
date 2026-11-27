"""
Keep the config.yml example of doc/docs/03_configurationFile.md in sync with the keys the code reads.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'doc' / 'docs' / '03_configurationFile.md'

# documented for the user, not read by the code
METADATA = {'project.name', 'project.description', 'project.date'}
# set by Topoclass from other keys, not read from config.yml
DERIVED = {'climate.path', 'climate.tmp_path', 'dem.filepath', 'outputs.path', 'outputs.tmp_path',
           'outputs.downscaled', 'sampling.pt_names'}
METHODS = {'lower', 'mkdir', 'get', 'exists', 'replace', 'glob', 'name', 'keys'}


def doc_block():
    return re.search(r'```yaml\n(.*?)```', DOC.read_text(), re.S).group(1)


def test_reference_config_is_the_documented_example():
    """config_reference.yml, used to check user configs, is the example of the documentation."""
    reference = (ROOT / 'TopoPyScale' / 'config_reference.yml').read_text()
    body = ''.join(line + '\n' for line in reference.splitlines() if not line.startswith('# Reference')
                   and not line.startswith('# Kept identical'))
    assert body == doc_block()


def doc_keys():
    config = yaml.safe_load(doc_block())

    def flatten(d, prefix=''):
        for k, v in d.items():
            key = f'{prefix}{k}'
            # values that are dictionaries of data, not of settings
            if isinstance(v, dict) and k != 'clustering_features':
                yield from flatten(v, key + '.')
            else:
                yield key
    return set(flatten(config))


def code_keys():
    keys = set()
    for f in (ROOT / 'TopoPyScale').glob('*.py'):
        src = f.read_text().replace('[self.config.project.climate]', '.era5')
        for line in src.splitlines():
            if line.strip().startswith('#'):
                continue
            for m in re.finditer(r'config((?:\.[A-Za-z_][A-Za-z0-9_]*)+)', line):
                parts = m.group(1).strip('.').split('.')
                while parts and parts[-1] in METHODS:
                    parts.pop()
                keys.add('.'.join(parts))
    return keys


def is_read(key, read):
    # a key is read if it, or a parent of it read as a whole (e.g. project.extent), is referenced
    parts = key.split('.')
    return any('.'.join(parts[:i]) in read for i in range(len(parts), 0, -1))


def test_documented_keys_are_read():
    read = code_keys()
    unread = sorted(k for k in doc_keys() - METADATA if not is_read(k, read))
    assert not unread, f'keys documented in 03_configurationFile.md but not read by the code: {unread}'


def test_read_keys_are_documented():
    documented = doc_keys()
    prefixes = {'.'.join(k.split('.')[:i]) for k in documented for i in range(1, len(k.split('.')))}
    missing = sorted(k for k in code_keys() - DERIVED
                     if k.split('.')[0] in {'project', 'climate', 'dem', 'sampling', 'toposcale', 'outputs', 'clean_up'}
                     and k not in documented and k not in prefixes)
    assert not missing, f'keys read by the code but missing from the config.yml example: {missing}'
