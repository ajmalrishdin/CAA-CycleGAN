"""Delete step-numbered discriminator checkpoints from a model directory.

Discriminators are only needed to continue training, and current runs keep them
in resume.pth instead of the step-numbered archive. Older directories still hold
one 16MB discriminator per module per saved step, which dominates their size.

Dry-run by default; pass --apply to actually delete.

    python tools/prune_discriminators.py models/V2.1
    python tools/prune_discriminators.py models/V2.1 --apply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.utils import (
    DISCRIMINATOR_SUFFIXES,
    RESUME_STATE_FILENAME,
    previous_generation_path,
)


def discriminator_files(model_dir):
    suffixes = tuple(f'_{suffix}.pth' for suffix in DISCRIMINATOR_SUFFIXES)
    return sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(suffixes) and name.split('_')[0].isdigit()
    )


def has_resume_state(model_dir):
    resume_path = os.path.join(model_dir, RESUME_STATE_FILENAME)
    return os.path.isfile(resume_path) or os.path.isfile(previous_generation_path(resume_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_dir', help='e.g. models/V2.1')
    parser.add_argument('--apply', action='store_true',
                        help='actually delete (default is a dry run)')
    parser.add_argument('--force', action='store_true',
                        help=f'allow deleting even when the directory has no {RESUME_STATE_FILENAME}')
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        parser.error(f'not a directory: {args.model_dir}')

    paths = discriminator_files(args.model_dir)
    total_bytes = sum(os.path.getsize(path) for path in paths)
    print(f'{len(paths)} discriminator files, {total_bytes / 1e9:.2f} GB in {args.model_dir}')
    if not paths:
        return

    if not args.apply:
        for path in paths[:10]:
            print(f'  would delete {os.path.basename(path)}')
        if len(paths) > 10:
            print(f'  ... and {len(paths) - 10} more')
        print('Dry run: nothing deleted. Re-run with --apply to delete.')
        return

    if not has_resume_state(args.model_dir) and not args.force:
        parser.error(
            f'{args.model_dir} has no {RESUME_STATE_FILENAME}, so these files are the only '
            f'discriminator weights for that run; deleting them means a resume restarts the '
            f'discriminators from scratch. Pass --force if that is acceptable.'
        )

    deleted = 0
    for path in paths:
        try:
            os.remove(path)
            deleted += 1
        except OSError as exc:
            print(f'  could not delete {path}: {exc}')
    print(f'Deleted {deleted} files, freed {total_bytes / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
