import zarr
import numpy as np
import pickle
import tqdm

z = zarr.open('garmentnets_images.zarr', mode='r')
keys = {}
for category in z:
    for instance in tqdm.tqdm(z[category]['samples'], desc=category):
        if category not in keys:
            keys[category] = []
        keys[category].append(instance)
        # instance_z = z[category]['samples'][instance]


# split the dataset
np.random.seed(0)
train_keys = []
test_keys = []
for cat in keys:
    np.random.shuffle(keys[cat])
    train_keys.extend([(cat,k) for k in keys[cat][:-100]])
    test_keys.extend([(cat,k) for k in keys[cat][-100:]])

with open('train.pkl','wb') as file:
    pickle.dump(train_keys, file)

with open('test.pkl','wb') as file:
    pickle.dump(test_keys, file)
