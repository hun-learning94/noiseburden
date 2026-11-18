"""Train or stage the four ImageNet128 models in the A1 comparison."""
import argparse
import copy
import json
from pathlib import Path
import sys

from evaluation.runtime import ROOT, SPEC, digest, save


def cases():
    return json.loads((ROOT / 'configs/rebuttal_width.json').read_text())


def setup(args):
    mapping_file = args.checkpoints.resolve()
    mapping = json.loads(mapping_file.read_text())
    selected = cases()
    for case in selected:
        if case['id'] not in mapping:
            raise ValueError(f"Missing checkpoint mapping for {case['id']}")
        path = Path(mapping[case['id']]).expanduser()
        if not path.is_absolute():
            path = mapping_file.parent / path
        actual = digest(path)
        if not args.accept_new_checkpoints and actual != case['checkpoint_sha256']:
            raise ValueError(f"Checkpoint checksum mismatch for {case['id']}")
        case['published_checkpoint_sha256'] = case['checkpoint_sha256']
        case['checkpoint_sha256'] = actual
        case['checkpoint'] = str(path.resolve())
        case['source'] = str(ROOT / case['source'])
        for name, expected in case['source_hashes'].items():
            assert digest(Path(case['source']) / name) == expected, name
    # Validate all paths/checksums before writing any staged model description.
    for case in selected:
        save(args.workspace / case['id'] / 'case.json', case)
    print(f'Staged {len(selected)} A1 configurations in {args.workspace}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Reuse the released effective-batch, EMA and local-cache implementation.
        # Keep the pinned arXiv configuration file and its original case set intact.
        known = {c['id'] for c in SPEC['cases']}
        SPEC['cases'].extend(copy.deepcopy(c) for c in cases() if c['id'] not in known)
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from training.run import main as train
        train()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['setup'])
    parser.add_argument('--workspace', type=Path, default=Path('work/a1'))
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--accept-new-checkpoints', action='store_true')
    setup(parser.parse_args())


if __name__ == '__main__':
    main()
