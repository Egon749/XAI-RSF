import shlex
import pytest
from experiments.post_st_dropcolumn_selection_h24_v1.publication_recovery import relay_command


def test_publication_relay_preserves_quoted_source_and_staging_only():
    r={'job':'2597','task':12,'info':{'output':'/home/lemon/work/model $draft'},'spec':{'results':'/home/oliver/MainDiskData/results'}}
    argv=['rsync','-rlt','--',r['info']['output']+'/',r['spec']['results']+'/2597/.partial-12-unique/output/']
    got=relay_command(argv,r)
    assert got[:4]==['ssh','-o','BatchMode=yes','oliver']
    assert shlex.split(got[4])==['rsync','-rlt','-s','--','lemon:/home/lemon/work/model $draft/',argv[4]]


def test_publication_relay_rejects_final_directory_or_changed_source():
    r={'job':'2597','task':12,'info':{'output':'/scratch/output'},'spec':{'results':'/results'}}
    for source,target in [('/scratch/output/','/results/2597/task-12/output/'),('/other/','/results/2597/.partial-12-unique/output/'),('/scratch/output/','/results/2597/.partial-13-unique/output/')]:
        with pytest.raises(ValueError):relay_command(['rsync','-rlt','--',source,target],r)
