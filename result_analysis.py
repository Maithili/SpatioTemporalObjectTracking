# %%
import os
import numpy as np
import json

# %%
dir_out = 'logspersona0329/visuals'
with open(os.path.join(dir_out,'info.json')) as f:
    info = json.load(f)
print(info)

# %%
print({kk:{k:sum(v) for k,v in vv.items()} for kk,vv in info.items()})
info_averages = {kk:{k:np.mean(v) for k,v in vv.items()} for kk,vv in info.items()}
info_mins = {kk:{k:min(v) for k,v in vv.items()} for kk,vv in info.items()}
info_maxs = {kk:{k:max(v) for k,v in vv.items()} for kk,vv in info.items()}
info_stds = {kk:{k:np.std(v) for k,v in vv.items()} for kk,vv in info.items()}

print(info_averages)
print(info_mins)
print(info_maxs)
print(info_stds)
methods = info.keys()

# %%
for res in ['precision', 'recall', 'destination_accuracy']:
    print('-----',res,'-----')
    for m in methods:
        print('{} : {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(m+' '*(25-len(m)), info_mins[m][res], info_averages[m][res]-info_stds[m][res], info_averages[m][res], info_averages[m][res]+info_stds[m][res], info_maxs[m][res]))



