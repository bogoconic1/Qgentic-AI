import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.researcher import _build_directory_listing

listing = _build_directory_listing("task/us-patent-phrase-to-phrase-matching")
print(listing)