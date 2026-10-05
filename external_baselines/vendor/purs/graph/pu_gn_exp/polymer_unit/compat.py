from pathlib import Path
import sys


def ensure_repo_root_on_path():
    current = Path(__file__).resolve()
    src_root = current.parents[4]
    repo_root = current.parents[5]
    for candidate in (src_root, repo_root):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return repo_root
