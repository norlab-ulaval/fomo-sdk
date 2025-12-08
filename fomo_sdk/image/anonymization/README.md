# Anonymization

This folder containes the code and the files to build apptainer containers to run EgoBlur models on all the images in a folder.

## Build the apptainer

```shell
apptainer build EgoBlur.sif EgoBlur.def
```

## Run the apptainer

```shell
apptainer run --nv -B ./:/mnt/egoblur EgoBlur.sif
```

## Run on mamba

```shell
sbatch --export=INPUT_PATH=input_path,OUTPUT_PATH=output_path slurm.sh
```

```
sbatch --export=INPUT_PATH=/mnt/bigfoot/FoMo/ijrr_temp/,OUTPUT_PATH=/mnt/bigfoot/FoMo/ijrr_final/ $HOME/fomo/fomo-sdk/fomo_sdk/image/anonymization/slurm.sh
```
