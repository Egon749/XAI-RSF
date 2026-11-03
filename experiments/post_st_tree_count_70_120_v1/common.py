from experiments.post_st_tree_count_local_v1.common import *
NAME='post_st_tree_count_70_120_v1'
REPO=Path(__file__).resolve().parents[2]
COUNTS=list(range(24,41))
NEW_COUNTS=[24,25,26]
def previous_path():return Path(load(REPO/'experiments/post_st_tree_count_local_v1/run_location.json')['private_run'])
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])

