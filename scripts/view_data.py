#%% IMPORTS

import os
import numpy as np
import sys
import DMNN_SGD.configuration as configuration

#%% Config

Config = configuration.Config()

#%% Definitions

def recursive_shape(data, depth=1,indent=6):
    if isinstance(data, np.ndarray):
        print(f'{" "*indent}{"--"*depth}> {data.shape}')
    else:
        print(f'{" "*indent}{"--"*depth}> {len(data)}')
        return recursive_shape(data[0],depth=depth+1)

def xray_npz(npz,indent=4):
    for key in npz.keys():
        print(f'{" "*indent}{key}')
        recursive_shape(npz[key],indent=indent+2)

#%% Print Data

for i,file in enumerate(os.listdir(Config.data_path)):
    print(f'({i}) {file}')
    file_path = os.path.join(Config.data_path, file)
    data = np.load(file_path)
    xray_npz(data)
    print('\n'*1)

# %%

data['val_labels'].sum()

# %%

data['val_labels'].size

# %%
