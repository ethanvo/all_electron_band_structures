#!/usr/bin/env python
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools import pyscf_ase, lattice
import sys
import h5py

##############################
# Create a "Cell"
##############################

cell = gto.Cell()
# Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
formula = str(sys.argv[1])
ase_atom = lattice.get_ase_atom(formula)
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell
cell.unit = 'B'
cell.basis = str(sys.argv[2])
cell.verbose = 7
cell.build()

##############################
#  K-point SCF 
##############################
kdensity = int(sys.argv[3])
kmesh = [kdensity, kdensity, kdensity] 
scaled_center=[0.0, 0.0, 0.0]
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
mymf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald")
mymf = mymf.density_fit()
ekrhf = mymf.kernel()

##############################
#  Plot bands 
##############################

import matplotlib
matplotlib.use('Agg') # Tell matplotlib not to use Xwindows backend
import matplotlib.pyplot as plt
from ase.dft.kpoints import get_bandpath

# Get band path information
npoints = 10
bandpath = get_bandpath('GX', ase_atom.cell, npoints=npoints)
xcoords, xspecial, labels = bandpath.get_linear_kpoint_axis()

# Get band energies along band path
band_kpts_scaled = bandpath.kpts
band_kpts_abs = cell.get_abs_kpts(band_kpts_scaled)

e_kn = []
dm = None
for kcenter in band_kpts_scaled:
    kpts = cell.make_kpts(kmesh, scaled_center=kcenter)
    mymf = scf.KRHF(cell, kpts=kpts, exxdiv='ewald').density_fit()
    escf = mymf.kernel(dm0=dm)
    dm = mymf.make_rdm1()
    e_kn.append(mymf.mo_energy[0])

print('Shape of e_kn:', np.shape(e_kn))
print("kpts scaled")
print(band_kpts_scaled)
print("kpts abs")
print(band_kpts_abs)

# Shift valence band maximum to zero

vbmax = -99
vbmax_k = 0
for idx, en in enumerate(e_kn):
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
        vbmax_k = idx
e_kn = [en - vbmax for en in e_kn]

e_kn_arr = np.array(e_kn)
cbmin_k = np.asarray(np.unravel_index(np.argmin(np.where(e_kn_arr > 0.0, e_kn_arr, np.inf), axis=None), e_kn_arr.shape))
print('Conduction band minimum index:', cbmin_k)

# Find the degeneracy of the valence band max and conduction band min.
vbmax_k_arr = np.array(e_kn[vbmax_k])
cbmin_k_arr = np.array(e_kn[cbmin_k[0]])

g_vbmax = vbmax_k_arr[np.abs(vbmax_k_arr) < 0.001].size
g_cbmin = cbmin_k_arr[np.abs(cbmin_k_arr - np.amin(np.where(cbmin_k_arr > 0.0, cbmin_k_arr, np.inf))) < 0.001].size

print('Valence band degeneracy:', g_vbmax)
print('Conduction band degeneracy:', g_cbmin)

# Plot band structure
au2ev = 27.21139
emin = (np.amin(e_kn_arr) - 1) * au2ev
emax = (np.amax(e_kn_arr)) * au2ev
ax = plt.figure(figsize=(5, 6), dpi=100).add_subplot()
nbands = cell.nao_nr()
for n in range(nbands):
    ax.plot(xcoords, [e[n]*au2ev for e in e_kn], color='#87CEEB')
for p in xspecial:
    ax.plot([p, p], [emin, emax], 'k-')
ax.plot([0, xspecial[-1]], [0, 0], 'k-')
ax.set_xticks(xspecial)
ax.set_xticklabels([f'${x}$' for x in labels])
ax.axis(xmin=0, xmax=xspecial[-1], ymin=emin, ymax=emax)
ax.set_xlabel('k-vector')
ax.set_ylabel('Energy (eV)')

plt.savefig(formula + '_' + cell.basis + '_k' + str(kdensity) + '_bands.png')

h5f = h5py.File(formula + '_' + cell.basis + '_k' + str(kdensity) + '.hdf5', 'w')
h5f.create_dataset('e_kn', data=e_kn_arr)
h5f.close()
