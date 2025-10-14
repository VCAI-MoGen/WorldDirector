# SCVideo
## Branch Details
**camera parameters**:

Note: You can specify the:
- **DIRECTOR architecture**, i.e. A: `incontext_director`, B: `adaln_director`, C: `ca_director`.
- **Subset**, `pure` (only simple camera trajectories) or `mixed` (simple and complex camera trajectories).
  
Note: We use C: 'ca_director'as primary architecture and 'mixed' as primary subset 

To launch training, run:
```
export PYTHONPATH=$(pwd)

HYDRA_FULL_ERROR=1 python src/train.py \
diffuser/network/module=ARCHITECTURE_NAME dataset.trajectory.set_name=SUBSET_NAME
```

To launch evaluation, run:
```
export PYTHONPATH=$(pwd)

HYDRA_FULL_ERROR=1 python src/evaluate.py
compnode.num_gpus=1 \
diffuser/network/module=ARCHITECTURE_NAME dataset.trajectory.set_name=SUBSET_NAME \
checkpoint_path=CHECKPOINT_PATH diffuser/network/module=rn_mdm diffuser.guidance_weight=GUIDANCE_WEIGHT
```

## Visualization

Note: We use .pkl files from ./et-data/smplh/ to render mesh 


<details><summary>Rerun visualization</summary>
&emsp;

To launch Rerun visualization script, run:
```
export PYTHONPATH=$(pwd)

python visualization/rerun_viz.py
```


</details>
<br>




