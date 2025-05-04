# experiments_2022
This repository was created to replicate results found in "Relatively few 'dominant' zones drive energy savings in setpoint adjustment experiments: Implications for energy efficiency", J. Neutel, C McMahon, A. Bonnafont, T. Troxell, SM. Benson, JA. de Chalendar. The publication is currently under review.

# Installation and use 
This repo can be installed by cloning the repo onto your machine, and then in the same folder as the ```pyproject.toml``` file running the terminal command ```pip install -e .```. This command will also automatically install the required packages for this repo, and thus we recommend running it in a new virtual environment of some kind (for example, using ```conda```). 

You will also need to update the paths in ```experiments_2022/src/experiments_2022/paths.py```

Data for this work can be found here: https://drive.google.com/drive/u/0/folders/1Q2CmFuzpeoS3Q56-0uYOofrEi4QS1GJN. You will need to download this data and place it where your ```DATASETS_PATH``` is pointing to. 

After completing these steps, results can be replicated by running the ```2022Analysis.ipynb``` notebook. Note that some of the operations in this notebook assume prior cells have been run, so it is best to run the notebook sequentially. 

# Citation
To come after the review process. 

# Abstract
In this work, we report on cooling setpoint adjustment experiments conducted in the Summer of 2022 in seven office buildings and three lab buildings on Stanford University’s campus, spanning ~88,000 m2 and ~1,600 zones in total. We find significant energy savings, up to 51% depending on the building, but driven by relatively few responding zones: 23% of zones across the entire testbed. We explain why few zones responded to setpoint changes, discussing connections to overcooling – an inefficiency endemic to US commercial buildings – and “zonal heterogeneity” – the idea that zones have innately different cooling needs. In light of observed zonal heterogeneity, we highlight some practical solutions to reduce consumption in high-demanding “dominant” zones and overcooling in low-demanding “dominated” zones. Our results suggest that solutions can be prioritized in relatively few zones to attain significant energy savings with a fraction of the time and cost.

