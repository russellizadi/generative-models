# generative-models

Create and environment based on the default `environment.yml`:
```
conda env create
```

To update the environmet after installing new packages:
```
conda env export --from-history | grep -v "prefix" > environment.yml
```
