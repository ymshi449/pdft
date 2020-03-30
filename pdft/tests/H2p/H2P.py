import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
import libcubeprop

psi4.set_output_file("H2P.psi4")
functional = 'svwn'
basis = 'cc-pv6z'
svdc = -3
reguc = -7
mu = -7
title = "H2p BT %i svd:%i " %(mu,svdc) + basis + functional
print(title)
Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -1 0 0
H  1 0 0
units bohr
symmetry c1
""")

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H  1 0 0
units bohr
symmetry c1
""")
Full_Molec =  psi4.geometry("""
nocom
noreorient
1 2
H  -1 0 0
H  1 0 0
units bohr
symmetry c1""")
Full_Molec.set_name("H2P")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 50,
    'DFT_RADIAL_POINTS':    28,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, basis, functional)
f1  = pdft.U_Molecule(Monomer_2,  basis, functional, omega=0.5, jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional, omega=0.5, jk=mol.jk)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)


# pdfter.find_vp_response(49, svd_rcond=10**svdc, regul_const=10**reguc, beta=0.1, a_rho_var=1e-7)

# pdfter.find_vp_densitydifference(21)
# pdfter.find_vp_response(21, svd_rcond=10**svdc, regul_const=10**reguc, beta=0.1, a_rho_var=1e-7)
hess, jac = pdfter.find_vp_response_1basis(14, svd_rcond=10**svdc, a_rho_var=1e-7)
# pdfter.find_vp_scipy_1basis(maxiter=24)
# # pdfter.find_vp_densitydifference(42, 1)
# jac, hess = pdfter.find_vp_response_grid(maxiter=1, hessian_data_type="float32")

f,ax = plt.subplots(1,1, dpi=210)
ax.set_ylim(-2, 1)
pdft.plot1d_x(pdfter.vp_grid, mol.Vpot, dimmer_length=2,
              title=title + str(pdfter.drho_conv[-1]), ax=ax, label="vp")
# pdft.plot1d_x(pdfter.vp_Hext_nad, mol.Vpot,ax=ax, label="Hext", ls='--')
# pdft.plot1d_x(pdfter.vp_xc_nad, mol.Vpot, ax=ax, label="xc", ls='--')
# pdft.plot1d_x(pdfter.vp_kin_nad, mol.Vpot, ax=ax, label="kin", ls='--')
ax.legend()
f.show()
plt.close(f)

# vp_grid_DD = mol.to_grid(pdfter.vp_last[0])
# vp_grid_WY = mol.to_grid(mol.A.np.dot(pdfter.vp[0]))
# f,ax = plt.subplots(1,1, dpi=210)
# ax.set_ylim(-1.2, 1)
# pdft.plot1d_x(pdfter.vp_grid, mol.Vpot, ax=ax, label="vp", color='black')
# pdft.plot1d_x(vp_grid_DD, mol.Vpot, ax=ax, label="vp_DD")
# pdft.plot1d_x(vp_grid_WY, mol.Vpot, dimmer_length=2,
#               title="vpDD+WY" + title + str(pdfter.drho_conv[-1]), ax=ax, label="vp_WY")
# ax.legend()
# f.show()
# f.savefig("vpDD+WY" + title)
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


#%%
# pdfter.ep_conv = np.array(pdfter.ep_conv)
# plt.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), 'o')
# plt.title("log dEp")
# plt.show()


# for svd in np.linspace(1, 7, 20):
#     pdfter.find_vp_response(163, svd_rcond=1e-4, regul_const=10**(-svd), beta=0.1)
#     vp_grid = mol.to_grid(pdfter.vp[0])
#     f, ax = plt.subplots(1, 1,figsize=(16,12), dpi=160)
#     pdft.plot1d_x(vp_grid, mol.Vpot, title="%.14f, Ep:% f, drho:% f" %(10**(-svd), pdfter.ep_conv[-1], pdfter.drho_conv[-1]), ax=ax)
#     f.savefig("H2+l" + str(int(svd*100)))
#     print("===============================================lambda=%f, Ep:% f, drho:% f" %(10**(-svd), pdfter.ep_conv[-1], pdfter.drho_conv[-1]))