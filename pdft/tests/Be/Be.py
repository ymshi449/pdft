import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

separation = 4.522
functional = 'svwn'
basis = 'cc-pvtz'
svdc = -2
reguc = -7
# title = "Be WuYang1b Yan Q[nf] v[nf] svdc%i reguc%i " %(svdc, reguc) + basis + functional
title = "Be WuYang inv-svd dnBT: %i" %svdc + basis + functional
print(title)

psi4.set_output_file("Be2.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Full_Molec.set_name("Be2")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, basis, functional)
f1  = pdft.U_Molecule(Monomer_2,  basis, functional, jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional, jk=mol.jk)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
mol.scf(100)
E, wfn = psi4.energy("B3LYP/"+basis, molecule=Full_Molec, return_wfn=True)
n_mol = mol.to_grid(mol.Da.np+mol.Db.np)
n_wfn = mol.to_grid(wfn.Da().np+wfn.Db().np)
mol.Da.np[:] = np.copy(wfn.Da().np)
mol.Db.np[:] = np.copy(wfn.Db().np)
mol.energy = E

pdfter.find_vp_projection(100, projection_method="Huzinaga")
# pdfter.find_vp_densitydifference(63, scf_method="Orthogonal")

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

w = mol.Vpot.get_np_xyzw()[-1]
S = mol.S.np
ortho = [np.dot(f1.Cocca.np.T, S.dot(f2.Cocca.np)),
         np.dot(f1.Coccb.np.T, S.dot(f2.Coccb.np))]
print("orthogonality", ortho)
pdfter.update_vp_EDA(update_object=True)

f,ax = plt.subplots(1,1, dpi=210)
ax.set_ylim(-2, 1)
ax.set_xlim(-10, 10)
pdft.plot1d_x(pdfter.vp_grid, pdfter.molecule.Vpot, dimmer_length=2,
         ax=ax, label="vp", color="black")
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


# #%% 1 basis 2D plot
# L = [4.0, 4.0, 2.0]
# D = [0.1, 0.1, 0.1]
# # Plot file
# O, N = libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# vp_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
# vp_cube = libcubeprop.compute_density(mol.wfn, O, N, D, npoints, points, nxyz, block, vp_psi4)
# f, ax = plt.subplots(1, 1, dpi=160)
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
# f, ax = plt.subplots(1, 1, dpi=160)
# p = ax.imshow(dn_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
# atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
# ax.set_title("dn" + title)
# f.colorbar(p, ax=ax)
# f.show()
# f.savefig("dn" + title)