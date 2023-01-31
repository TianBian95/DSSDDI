# DSSDDI

## Package Dependency
Here is a list of reference versions for all package:
```
  python: 3.6.8
  dgl: 0.6.0
  numpy: 1.19.5
  pandas: 1.1.5
  torch: 1.7.1
  scipy: 1.5.4
  sklearn: 0.0.post1
```

### Step 1: Prepare KG files.
Download DRKG_TransE_l2_entity.npy from [DRKG](https://github.com/gnn4dr/DRKG) and move it to /data directory.

### Step 2: Generate treatment files.
Generate treatment by running:

``python3 ./causal_model/Treatment.py``

### Step 3: Generate counterfactual links files.
Generate counterfactual links by running:

``python3 ./causal_model/CounterfactualLinks.py``

### Step 4: Train and evaluate DSSDDI:
``python3 ./causal_model/main.py``
