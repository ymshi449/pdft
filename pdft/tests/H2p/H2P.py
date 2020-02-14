import psi4
import pdft
import matplotlib.pyplot as plt
import numpy as np
import libcubeprop

psi4.set_output_file("H2P.psi4")
functional = 'svwn'
basis = 'cc-pvdz'

Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -1 0 0
@H   0 0.5 0
@H   0 -0.5 0
H  1 0 0
units bohr
symmetry c1
""")

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H   0 0.5 0
@H   0 -0.5 0
@H  1 0 0
units bohr
symmetry c1
""")
Full_Molec =  psi4.geometry("""
1 2
nocom
noreorient
H  -1 0 0
@H   0 0.5 0
@H   0 -0.5 0
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


# pdfter.find_vp_response(50, svd_rcond=1e-4, regul_const=1e-3, beta=0.1, a_rho_var=1e-7)
pdfter.find_vp_response_1basis(21, svd_rcond=1e-2, regul_const=None, beta=0.1, a_rho_var=1e-7)
# pdfter.find_vp_scipy(maxiter=7, regul_const=1e-4)
# pdfter.find_vp_scipy_1basis(maxiter=21)

vp_grid = mol.to_grid_1basis(pdfter.vp[0])
pdft.plot1dx(vp_grid, mol.Vpot, title="H2+" + basis + functional)

#%% 1 basis 2D plot
L = [2.0, 2.0, 2.0]
D = [0.05, 0.1, 0.1]
# Plot file
O, N = libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
vp_cube = libcubeprop.compute_density_1basis(mol.wfn, O, N, D, npoints, points, nxyz, block, pdfter.vp[0])
f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
p = ax.imshow(vp_cube[40, :, :], interpolation="bicubic", cmap="Spectral")
atoms = libcubeprop.get_atoms(mol.wfn, D, O)
ax.scatter(atoms[:,3], atoms[:,2])
ax.set_title("vp H2p_svd1e-3_1b" + basis + functional)
f.colorbar(p, ax=ax)
f.show()
# f.savefig("H2p_svd1e-3_1b")


#%%
# pdfter.ep_conv = np.array(pdfter.ep_conv)
# plt.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), 'o')
# plt.title("log dEp")
# plt.show()
#
# for svd in np.linspace(1, 7, 20):
#     pdfter.find_vp_response(163, svd_rcond=1e-4, regul_const=10**(-svd), beta=0.1)
#     vp_grid = mol.to_grid(pdfter.vp[0])
#     f, ax = plt.subplots(1, 1,figsize=(16,12), dpi=160)
#     pdft.plot1d_x(vp_grid, mol.Vpot, title="%.14f, Ep:% f, drho:% f" %(10**(-svd), pdfter.ep_conv[-1], pdfter.drho_conv[-1]), ax=ax)
#     f.savefig("H2+l" + str(int(svd*100)))
#     print("===============================================lambda=%f, Ep:% f, drho:% f" %(10**(-svd), pdfter.ep_conv[-1], pdfter.drho_conv[-1]))