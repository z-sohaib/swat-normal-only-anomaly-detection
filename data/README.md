# SWaT Data

Place the SWaT CSV files here:

```text
data/raw/SWaT_Dataset_Normal_v1.csv
data/raw/SWaT_Dataset_Attack_v0.csv
```

The loader builds sliding windows from the normal and attack files. The default
config uses:

- window size: `100`
- stride: `50`
- label column: `Normal/Attack`

The `raw/` and `processed/` folders are ignored by Git.
