import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 4.522
functional = 'svwn'
basis = 'cc-pvdz'
svdc = -5
reguc = -4
title = "Be WuYang1b 0ghost svdc%i reguc%i" %(svdc, reguc) + basis + functional
print(title)

psi4.set_output_file("Be2.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Full_Molec.set_name("Be2")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  basis, functional)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional)
mol = pdft.U_Molecule(Full_Molec, basis, functional)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)

# pdfter.find_vp_response(21, svd_rcond=10**svdc, regul_const=10**reguc, beta=0.1, a_rho_var=1e-7)
pdfter.find_vp_response_1basis(21, svd_rcond=10**svdc, beta=1, a_rho_var=1e-7)
# pdfter.find_vp_scipy_1basis(maxiter=7)
# pdfter.find_vp_densitydifference(42, 1)


f,ax = plt.subplots(1,1)
ax.set_ylim(-2,1)
vp_grid = mol.to_grid_1basis(pdfter.vp[0])
pdft.plot1d_x(pdfter.vp_Hext_nad, mol.Vpot, title=title, ax=ax)
pdft.plot1d_x(vp_grid, mol.Vpot, title=title, ax=ax)
pdft.plot1d_x(pdfter.vp_Hext_nad + vp_grid, mol.Vpot, title=title, ax=ax)
f.show()

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