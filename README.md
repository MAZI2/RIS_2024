# RIS_2024
## Priprava okolja za Sling
```
module load Anaconda3
conda create -n env
source activate env

# za image_slicing.py
pip install nibabel scipy numpy tensorflow keras matplotlib opencv-python
```
sbatch_script.sh
```
#!/bin/bash
#SBATCH --job-name=image_slicing
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn01,gwn02,gwn03,gwn04,gwn05,gwn06,gwn07,gwn08
#SBATCH --output=result_slicing.txt 
#SBATCH --ntasks=1 
#SBATCH --nodes=1 
#SBATCH --time=1-00:00:00 
#SBATCH --gpus=1 
#SBATCH --mem-per-cpu=120GB

module load CUDA/10.1.105-GCC-8.2.0-2.31.1
module load Anaconda3/2023.07-2
source activate testEnv

conda run -n testEnv python image_slicing.py
```
oddaj posel
```
sbatch sbatch_script.sh
```
