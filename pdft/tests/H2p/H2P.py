import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
import libcubeprop

psi4.set_output_file("H2P.psi4")
functional = 'svwn'
basis = 'cc-pvtz'
svdc = -4
reguc = -7
title = "H2p BT" + basis + functional
print(title)
Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -1 0 0
@H 0 0 0
H  1 0 0
units bohr
symmetry c1
""")

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H 0 0 0
@H  1 0 0
units bohr
symmetry c1
""")
Full_Molec =  psi4.geometry("""
nocom
noreorient
1 2
H  -1 0 0
@H 0 0 0
H  1 0 0
units bohr
symmetry c1""")
Full_Molec.set_name("H2P")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  basis, functional, omega=0.5)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional, omega=0.5)
mol = pdft.U_Molecule(Full_Molec, basis, functional)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
mol.scf(100)
E, wfn = psi4.energy("B3LYP/"+basis, molecule=Full_Molec, return_wfn=True)
n_mol = mol.to_grid(mol.Da.np+mol.Db.np)
n_wfn = mol.to_grid(wfn.Da().np+wfn.Db().np)
mol.Da.np[:] = np.copy(wfn.Da().np)
mol.Db.np[:] = np.copy(wfn.Db().np)

pdfter.find_vp_densitydifference(350, scf_method="Orthogonal", beta_initial=0.2)

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
nf = 0.5*(n1 + n2)
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

# mol.scf(100)
# n1 = pdfter.molecule.to_grid(f1.Da.np + f1.Db.np)
# n2 = pdfter.molecule.to_grid(f2.Da.np + f2.Db.np)
# nf = mol.to_grid(pdfter.fragments_Db+pdfter.fragments_Da)
# n_mol = mol.to_grid(mol.Da.np+mol.Db.np)
# f,ax = plt.subplots(1,1, dpi=210)
# ax.set_ylim(-1, 0.5)
# ax.set_xlim(-10, 10)
# pdft.plot1d_x(pdfter.vp_Hext_nad + pdfter.vp_xc_nad, pdfter.molecule.Vpot, dimmer_length=2,
#          ax=ax, label="vp", color="black")
# pdft.plot1d_x(nf, pdfter.molecule.Vpot, ax=ax, label="nf", ls="--")
# pdft.plot1d_x(n_mol, pdfter.molecule.Vpot, ax=ax, label="nmol")
# pdft.plot1d_x(n1*0.5, pdfter.molecule.Vpot, ax=ax, label="n1", ls="dotted")
# pdft.plot1d_x(n2*0.5, pdfter.molecule.Vpot, ax=ax, label="n2", ls="dotted")
# pdft.plot1d_x(pdfter.vp_Hext_nad, pdfter.molecule.Vpot,
#          ax=ax, label="vpHext", ls='--')
# pdft.plot1d_x(pdfter.vp_xc_nad, pdfter.molecule.Vpot,
#          ax=ax, label="vpxc", ls='--')
# ax.legend()
# f.show()
# plt.close(f)

# #%% 1 basis 2D plot
# L = [2.0, 2.0, 2.0]
# D = [0.1, 0.1, 0.1]
# # Plot file
# O, N = libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# vp_cube = libcubeprop.compute_density_1basis(mol.wfn, O, N, D, npoints, points, nxyz, block, pdfter.vp[0])
# f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
# p = ax.imshow(vp_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
# atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
# ax.set_title("vp" + title)
# f.colorbar(p, ax=ax)
# f.show()
# f.savefig("vp" + title)
#
# dD = psi4.core.Matrix.from_array(pdfter.fragments_Da + pdfter.fragments_Db - mol.Da.np - mol.Db.np)
# dn_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, dD)
# f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
# p = ax.imshow(dn_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
# atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
# ax.set_title("dn" + title)
# f.colorbar(p, ax=ax)
# f.show()
# f.savefig("dn" + title)

# #%% 1 basis 2D plot
L = [4.0, 4.0, 2.0]
D = [0.1, 0.1, 0.1]
# Plot file
O, N = libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
D_mol = psi4.core.Matrix.from_array(mol.Da.np+mol.Db.np)
D_f = psi4.core.Matrix.from_array(pdfter.fragments_Da+pdfter.fragments_Db)
n_mol_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, D_mol)
n_f_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, D_f)

f, ax = plt.subplots(1, 1, dpi=160)
p = ax.imshow(n_mol_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
atoms = libcubeprop.get_atoms(mol.wfn, D, O)
ax.scatter(atoms[:,2], atoms[:,1])
ax.set_title("nmol" + title)
f.colorbar(p, ax=ax)
f.show()

f, ax = plt.subplots(1, 1, dpi=160)
p = ax.imshow(n_f_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
atoms = libcubeprop.get_atoms(mol.wfn, D, O)
ax.scatter(atoms[:,2], atoms[:,1])
ax.set_title("nf" + title)
f.colorbar(p, ax=ax)
f.show()

f, ax = plt.subplots(1, 1, dpi=160)
p = ax.imshow((n_mol_cube-n_f_cube)[:, :, 20], interpolation="bicubic", cmap="Spectral")
atoms = libcubeprop.get_atoms(mol.wfn, D, O)
ax.scatter(atoms[:,2], atoms[:,1])
ax.set_title("dn" + title)
f.colorbar(p, ax=ax)
f.show()

#%%
vp_fock_Hextxc = mol.grid_to_fock(pdfter.vp_Hext_nad + pdfter.vp_Hext_nad)
f1.scf(1000)
f2.scf(1000)
for i in range(14):
    coeff = 0.1 * i
    if coeff > 1:
        coeff = 1
    vp_fock_psi4 = psi4.core.Matrix.from_array(coeff * vp_fock_Hextxc)
    pdfter.orthogonal_scf(1000, vp_matrix=[vp_fock_psi4, vp_fock_psi4],
                          projection_method="Huzinaga")
    nftemp = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)
    dntemo = np.sum(np.abs(n_mol - nftemp) * w)
    Tsnad = abs(np.trace((mol.Da.np - pdfter.fragments_Da).dot(mol.T.np)))
    print("%.2f"%coeff, dntemo, Tsnad)