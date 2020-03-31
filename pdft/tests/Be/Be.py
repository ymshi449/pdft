import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

separation = 4.522
functional = 'svwn'
basis = 'cc-pvqz'
svdc = -4
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

# @Kr 0 0 7
# @Kr 0 0 -7
# @Kr 0 7 0
# @Kr 0 -7 0
# @Kr 7 0 0
# @Kr -7 0 0

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
# pdfter.find_vp_densitydifference(140)
# pdfter.find_vp_response(21, guess=True, svd_rcond=10**svdc, beta=0.1, a_rho_var=1e-7)
hess, jac = pdfter.find_vp_response_1basis(1, svd_rcond=10**svdc, a_rho_var=1e-5)
# pdfter.find_vp_scipy_1basis(maxiter=7)
#
# f,ax = plt.subplots(1,1, dpi=210)
# ax.set_ylim(-1.2, 0.2)
# # ax.set_xlim(-10, 10)
# pdft.plot1d_x(pdfter.vp_grid, mol.Vpot, dimmer_length=separation,
#               title="vp" + title + str(pdfter.drho_conv[-1]), ax=ax, label="vp", color='black')
# # pdft.plot1d_x(pdfter.vp_Hext_nad, mol.Vpot,  ax=ax, label="Hext", ls='--')
# # pdft.plot1d_x(pdfter.vp_xc_nad, mol.Vpot, ax=ax, label="xc", ls='--')
# # pdft.plot1d_x(pdfter.vp_kin_nad, mol.Vpot, ax=ax, label="kin", ls='--')
# ax.legend()
# f.show()
# f.savefig("vp" + title)
# plt.close(f)

# vp_grid_DD = mol.to_grid(pdfter.vp_last[0])
# vp_grid_WY = mol.to_grid(pdfter.vp[0])
# f,ax = plt.subplots(1,1, dpi=210)
# ax.set_ylim(-1.2, 0.2)
# pdft.plot1d_x(pdfter.vp_grid, mol.Vpot, ax=ax, label="vp", color='black')
# pdft.plot1d_x(vp_grid_DD, mol.Vpot, ax=ax, label="vp_DD")
# pdft.plot1d_x(vp_grid_WY, mol.Vpot, dimmer_length=separation,
#               title="vpDD+WY" + title + str(pdfter.drho_conv[-1]), ax=ax, label="vp_WY")
# ax.legend()
# f.show()
# f.savefig("vpDD+WY" + title)
# plt.close(f)
#
# # 1D density differnece
# nf = mol.to_grid(pdfter.fragments_Db + pdfter.fragments_Da)
# nmol = mol.to_grid(mol.Da.np + mol.Db.np)
# f,ax = plt.subplots(1,1)
# # pdft.plot1d_x(nmol, mol.Vpot, dimmer_length=separation, ax=ax)
# # pdft.plot1d_x(nf, mol.Vpot, dimmer_length=separation, ax=ax)
# pdft.plot1d_x(nmol - nf, mol.Vpot, dimmer_length=separation, ax=ax)
# f.show()
# plt.close(f)

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