import numpy as np
import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop

psi4.core.set_output_file("formic.psi4")
functional = 'svwn'
basis = 'cc-pvdz'
svdc = -4
reguc = -4
title = "formic Newton svdc%i reguc %i" %(svdc, reguc) + basis + functional
print(title)

Full_Molec = psi4.geometry("""
nocom
noreorient
C    0.0000000    0.1929272   -1.9035340
O    0.0000000    1.1595219   -1.1616236
O    0.0000000   -1.0680669   -1.5349870
H    0.0000000    0.2949802   -2.9949776
H    0.0000000   -1.1409414   -0.5399614
C    0.0000000   -0.1929272    1.9035340
O    0.0000000   -1.1595219    1.1616236
O    0.0000000    1.0680669    1.5349870
H    0.0000000   -0.2949802    2.9949776
H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Monomer_1 = psi4.geometry("""
nocom
noreorient
@C    0.0000000    0.1929272   -1.9035340
@O    0.0000000    1.1595219   -1.1616236
@O    0.0000000   -1.0680669   -1.5349870
@H    0.0000000    0.2949802   -2.9949776
@H    0.0000000   -1.1409414   -0.5399614
C    0.0000000   -0.1929272    1.9035340
O    0.0000000   -1.1595219    1.1616236
O    0.0000000    1.0680669    1.5349870
H    0.0000000   -0.2949802    2.9949776
H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Monomer_2 = psi4.geometry("""
nocom
noreorient
C    0.0000000    0.1929272   -1.9035340
O    0.0000000    1.1595219   -1.1616236
O    0.0000000   -1.0680669   -1.5349870
H    0.0000000    0.2949802   -2.9949776
H    0.0000000   -1.1409414   -0.5399614
@C    0.0000000   -0.1929272    1.9035340
@O    0.0000000   -1.1595219    1.1616236
@O    0.0000000    1.0680669    1.5349870
@H    0.0000000   -0.2949802    2.9949776
@H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Full_Molec.set_name("Large")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 434,
    # 'DFT_RADIAL_POINTS': 99,
    'REFERENCE' : 'UKS'})

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, basis, functional)
f1  = pdft.U_Molecule(Monomer_2,  basis, functional, jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional, jk=mol.jk)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
mol.scf(1000)
E, wfn = psi4.energy("B3LYP/"+basis, molecule=Full_Molec, return_wfn=True)
n_mol = mol.to_grid(mol.Da.np+mol.Db.np)
n_wfn = mol.to_grid(wfn.Da().np+wfn.Db().np)
mol.Da.np[:] = np.copy(wfn.Da().np)
mol.Db.np[:] = np.copy(wfn.Db().np)

pdfter.find_vp_densitydifference(35, scf_method="Orthogonal")

assert np.allclose(wfn.Da().np, mol.Da.np)

D1a = np.copy(f1.Da.np)
D2a = np.copy(f2.Da.np)
D1b = np.copy(f1.Db.np)
D2b = np.copy(f2.Db.np)
Cocc1a = np.copy(f1.Cocca.np)
Cocc2a = np.copy(f2.Cocca.np)
Cocc1b = np.copy(f1.Coccb.np)
Cocc2b = np.copy(f2.Coccb.np)

n1 = pdfter.molecule.to_grid(f1.Da.np + f1.Db.np)
n2 = pdfter.molecule.to_grid(f2.Da.np + f2.Db.np)
nf = n1 + n2
n_mol = mol.to_grid(mol.Da.np+mol.Db.np)

pdfter.update_vp_EDA(update_object=True)

w = mol.Vpot.get_np_xyzw()[-1]
S = mol.S.np
ortho = [np.dot(f1.Cocca.np.T, S.dot(f2.Cocca.np)),
         np.dot(f1.Coccb.np.T, S.dot(f2.Coccb.np))]
print("orthogonality", ortho)

f,ax = plt.subplots(1,1, dpi=210)
ax.set_ylim(-2, 1)
ax.set_xlim(-10, 10)
pdft.plot1d_x(pdfter.vp_grid, pdfter.molecule.Vpot, dimmer_length=2,
         ax=ax, label="vp", title=mol.wfn.molecule().name(),color="black")
pdft.plot1d_x(nf, pdfter.molecule.Vpot, ax=ax, label="nf", ls="--")
pdft.plot1d_x(n_mol, pdfter.molecule.Vpot, ax=ax, label="nmol")
pdft.plot1d_x(pdfter.vp_Hext_nad, pdfter.molecule.Vpot,
         ax=ax, label="vpHext", ls=':')
pdft.plot1d_x(pdfter.vp_xc_nad, pdfter.molecule.Vpot,
         ax=ax, label="vpxc", ls=':')
pdft.plot1d_x(pdfter.vp_kin_nad, pdfter.molecule.Vpot,
         ax=ax, label="vpkin", ls=':')
ax.legend()
f.show()
plt.close(f)

#%%