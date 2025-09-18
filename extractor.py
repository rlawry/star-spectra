import numpy as np
import json
import re
from collections import OrderedDict

# Step 1: spectral type list (from your original script)
sptypes = ['O5','O7','O8','O9'] \
          + ['%s%i' % (k,i) for k in 'B' for i in range(1,10)] \
          + ['A0','A1','A2'] + ['A%i_+0.0_Dwarf' % i for i in range(3,10)] \
          + ['%s%i_+0.0_Dwarf' % (k,i) for k in 'FGKM' for i in range(1,10)]
sptypes[-1] = 'M9'
sptypes += ['L0','L1','L2','L3','L6']
for i in [4,5,8]: sptypes.remove('A%i_+0.0_Dwarf' % i)
for i in [6,8,9]: sptypes.remove('K%i_+0.0_Dwarf' % i)
sptypes.remove('B7')

# Step 2: load the .npy data
data_array = np.load("data/boss_sptypes.npy", allow_pickle=True)

# Sanity check
if len(data_array) != len(sptypes):
    raise ValueError(f"Length mismatch: {len(data_array)} spectra vs {len(sptypes)} types")

# Step 3: custom sorting for spectral type
def spectral_sort_key(sptype):
    match = re.match(r'([OBAFGKML])(\d*)', sptype)
    if not match:
        return (999, 0)  # unknown types go at end
    letter, number = match.groups()
    # Define order: L coolest, O hottest
    letter_order = {'L':0, 'M':1, 'K':2, 'G':3, 'F':4, 'A':5, 'B':6, 'O':7}
    num = int(number) if number else 0
    return (letter_order[letter], num)

# Step 4: build spectra_dict with downsampling
spectra_pairs = []
step = 10  # downsample factor

for sptype, spectrum in zip(sptypes, data_array):
    wl = 10 ** spectrum['LogLam'] / 10.0
    flux = spectrum['Flux']
    wl_ds = wl[::step]
    flux_ds = flux[::step]
    spectra_pairs.append((sptype, {
        "wavelength": wl_ds.tolist(),
        "flux": flux_ds.tolist()
    }))

# Step 5: sort pairs by spectral type
spectra_pairs_sorted = sorted(spectra_pairs, key=lambda x: spectral_sort_key(x[0]))

# Step 6: build an OrderedDict for JSON
spectra_ordered = OrderedDict(spectra_pairs_sorted)

# Step 7: dump to JSON
with open("spectra.json", "w") as f:
    json.dump(spectra_ordered, f)

print(f"Saved {len(spectra_ordered)} spectra in astrophysical order (L → O).")
