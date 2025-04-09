

import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import debugpy
import os

debugpy.listen(('0.0.0.0', 5678))
print("Waiting for debugger attach")
debugpy.wait_for_client()


folder = 'batch_embedding_save_val'
embedding_batches = os.listdir(f'/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/{folder}')
n_batches = len(embedding_batches)


y = []
for b in range(n_batches):
    for i in range(16):
        y.append(i)   
        
        
all_embeddings = np.zeros((n_batches*16, 512))
for idx, batch_path in enumerate(embedding_batches):
    with open(f'{folder}/{batch_path}', 'rb') as f:
        cond_embedding = np.load(f)
    all_embeddings[idx*16:(idx*16) + 16] = cond_embedding

transformed = TSNE(n_components=2, perplexity=50.0,random_state=0).fit_transform(all_embeddings)

y_unsqueeze = np.expand_dims(y, axis=-1)
data = pd.DataFrame(np.concatenate((transformed, y_unsqueeze), axis=-1))
# pd.DataFrame(np.concatenate((transformed, y_unsqueeze), axis=-1))

import colorcet as cc
palette = sns.color_palette(cc.glasbey, n_colors=16)
plt.figure()
ax = sns.scatterplot(
    x=0, y=1,
    hue=2,
    palette=palette,
    data=data,
    legend="full",
    # alpha=0.3
)
plt.savefig(f'scatter_all_embedding_sorted_val_2.png')