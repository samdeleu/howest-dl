from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
#%%
#-----------------------
SESSIE_02_PATH = Path(__file__).parent.parent.parent.parent / Path("sessie_02/demos")
# Inlezen image
base_image = plt.imread(SESSIE_02_PATH / "Tesla.png")
#-----------------------
#%%
fig = plt.figure(figsize=(5,2.5))
print(plt.imshow(base_image))
#%%

if __name__ == '__main__':
    print("main")