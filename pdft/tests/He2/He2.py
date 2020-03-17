import sys
sys.path.append('../')
import scipy.ndimage
import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

separation = 6.0
functional = 'svwn'
basis = 'aug-cc-pvdz'
svdc = -7
reguc = -2
title = "He2 WuYang1b svd%i regu%i" %(svdc, reguc) + basis + functional
print(title)

psi4.set_output_file("He2")

Full_Molec =  psi4.geometry( """
nocom
noreorient
He %f 0.0 0.00
He -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
He %f 0.0 0.00
@He -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@He %f 0.0 0.00
He -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

Full_Molec.set_name("He2")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

# psi4.set_options({'cubeprop_tasks' : ['density'],
#                  'cubic_grid_spacing': [0.1, 0.1, 0.1]})

# energy_3, wfn_3 = psi4.energy("SVWN/cc-pvdz", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, basis, functional)
f1  = pdft.U_Molecule(Monomer_2,  basis, functional, jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  basis, functional, jk=mol.jk)

#Start a pdft systemm
pdfter = pdft.U_Embedding([f1, f2], mol)

#%% Running SCF with svwn
pdfter.find_vp_response_1basis(49, vp_nad_iter=None,
                               svd_rcond=10**svdc,
                               # regul_const=None,
                               a_rho_var=1e-7)

f,ax = plt.subplots(1,1, dpi=210)
pdft.plot1d_x(pdfter.vp_grid, mol.Vpot, ax=ax, label="vp", color='black')
pdft.plot1d_x(pdfter.vp_Hext_nad, mol.Vpot, dimmer_length=separation,
              title=title, ax=ax, label="Hext", ls='--')
pdft.plot1d_x(pdfter.vp_xc_nad, mol.Vpot, ax=ax, label="xc", ls='--')
pdft.plot1d_x(pdfter.vp_kin_nad, mol.Vpot, ax=ax, label="kin", ls='--')
ax.legend()
f.show()
plt.close(f)
# #%% Get vp_all96
# vp_all96, vp_fock_all96 = pdfter.vp_all96()
# print("vp all96", vp_fock_all96)
# pdft.plot1d_x(vp_all96, mol.Vpot, title="vp all96:" + str(separation), fignum=2)
# #%% Running SCF with vp_all96
# vp_fock_psi4 = psi4.core.Matrix.from_array(vp_fock_all96)
# pdfter.fragments_scf(max_iter=1000, vp=True, vp_fock=[vp_fock_psi4, vp_fock_psi4])
# n_all96 = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)
# pdft.plot1d_x(n_all96 - n_novp, mol.Vpot, title="density difference all96 - novp" + str(separation), fignum=3)
# pdft.plot1d_x(n_all96 - n_svwn, mol.Vpot, title="density difference all96 - svwn" + str(separation), fignum=4)
# #%% Campare density difference
# w = mol.Vpot.get_np_xyzw()[-1]
# print("==========DENSITY DIFFERENCE==========")
# print("Density difference no vp and all96.", np.sum(np.abs(n_novp - n_all96)*w))
# print("Density difference svwn and all96.", np.sum(np.abs(n_all96 - n_svwn)*w))
# print("Density difference no vp and svwn.", np.sum(np.abs(n_novp - n_svwn)*w))
#
# #%% Check if vp_all96 consist with vp_fock_all96
# if pdfter.four_overlap is None:
#     pdfter.four_overlap, _, _, _ = pdft.fouroverlap(pdfter.molecule.wfn, pdfter.molecule.geometry,
#                                                     pdfter.molecule.basis, pdfter.molecule.mints)
# vp_basis_all96 = mol.to_basis(vp_all96)
# vp_fock_all96_1 = np.einsum("abcd, ab -> cd", pdfter.four_overlap, vp_basis_all96)
# print("vp_all96 consists with vp_fock_all96?", np.allclose(vp_fock_all96_1, vp_fock_all96, atol=np.linalg.norm(vp_fock_all96)*0.1))
# vp_all96_1 = mol.to_grid(vp_basis_all96)
# print("basis and grid back and forth consistence?", np.allclose(vp_all96_1, vp_all96, atol=np.linalg.norm(vp_all96)*0.1))
# if not np.allclose(vp_all96_1, vp_all96, atol=np.linalg.norm(vp_all96)*0.1):
#     print("size of basis", mol.nbf)
#     print("The difference is", np.linalg.norm(vp_all96_1 - vp_all96)/np.linalg.norm(vp_all96))
#     pdft.plot1d_x(vp_all96_1, mol.Vpot, title="vp presented by the basis", fignum=5)
# # %% Plot vp_all96 on grid.
# L = [5.0,  5.0, 4.0]
# D = [0.2, 0.2, 0.2]
# # Plot file
# O, N = libcubeprop.build_grid(mol.wfn, L, D)
# block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
# # vp_basis_all96_psi4 = psi4.core.Matrix.from_array(pdfter.vp[0])
# vp_cube = libcubeprop.compute_density_1basis(mol.wfn, O, N, D, npoints, points, nxyz, block, pdfter.vp[0])
# f, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=160)
# plt.imshow(vp_cube[:,:,20], interpolation="bicubic")
# # plt.title("vpALL96 on basis.")
# plt.colorbar(fraction=0.040, pad=0.04)
# f.show()
# plt.savefig("vp2D")
