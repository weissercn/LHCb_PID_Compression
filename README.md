# LHCb_PID_Compression
A benchmark to the challenge of compressing an obscured dataset containing Particle Identification

GANs and Flows are trained in the exp\_3\_features directory with jupyter notebooks. Once run, the models and preprocessing scalers are saved in exp\_3\_features\weights and exp\_3\_features\gan\_preprocessors, respectively. 
Once trained the models can be accessed by running scripts like generate\_Flow\_gencut\_ksttrain\_nspd.py. 

Both the data provided for running, and the processing of the generated data is done with the following repo https://github.com/weissercn/LUV_ML and the scriot Efficiency.py within it.  
For individuals in the Mike Williams group at MIT, you can access the data in exp\_3\_features/data in the following directory of geno:  /data/weisser\_genmodels/LHCb\_PID\_Compression\_exp\_3\_features\_data.


OLD INSTRUCTIONS:

The data can be found here: https://zenodo.org/record/1231531#.WyZSQFOFO3V

Download it and put it into a folder called 'Data' in this directory.

Installing FunctionScaler(https://github.com/weissercn/FunctionScaler/) as a dependency is required. This can be done using pip (pip install FunctionScaler). 
Other requirements: keras, sklearn, pandas, matplotlib, pickle

Run the following notebook in this order:
1. Prepare
2. Train
3. Analyse Output
4. Cross Check
5. ROOT compression




