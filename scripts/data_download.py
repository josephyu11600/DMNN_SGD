#%% IMPORT
import kagglehub
import os
import sys
sys.path.append('../')
import DMNN_SGD.configuration as configuration

#%% Config

Config = configuration.Config()

#%% DOWNLOAD

path = kagglehub.dataset_download("adibadea/chbmitseizuredataset")
print("Path to dataset files:", path)

# %% MOVE

os.system(f'mv {path}/* {Config.data_path}')

# %%
