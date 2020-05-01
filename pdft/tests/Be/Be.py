import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

separation = 4.522
functional = 'svwn'
# There are two possibilities why a larger basis set is better than a smaller one:
# 1. It provides a larger space for the inverse of Hessian.
# 2. Or more likely it provides a more MOs for a better description of Hessian first order approximation.
basis = 'cc-pvdz'
vp_basis = 'cc-pvqz'
svdc = 1e-4
regulc = None
Orthogonal_basis = False
lag_tap = 1
scipy_method = "trust-exact"
# title = "Be WuYang1b Yan Q[nf] v[nf] svdc%i reguc%i " %(svdc, reguc) + basis + functional
title = F"ortho_vp_basis svd {svdc} regu {regulc} mol_basis {basis} vp_basis {vp_basis} " \
        + functional + " orth_basis: " + str(Orthogonal_basis)
print(title)
print(scipy_method, lag_tap)

psi4.set_output_file("Be2.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))

# @Ar 0 7 0
# @Ar 0 -7 0
# @Ar 0 0 7
# @Ar 0 0 -7

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

# Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, basis, functional)
f1 = pdft.U_Molecule(Monomer_2, basis, functional, jk=mol.jk)
f2 = pdft.U_Molecule(Monomer_1, basis, functional, jk=mol.jk)
if vp_basis is not None:
    vp_tester = pdft.U_Molecule(Full_Molec, vp_basis, functional)
    vp_tester.scf(100)
else:
    vp_tester = mol

# Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol, vp_basis=vp_basis)
pdfter.Lagrange_mul = lag_tap
# pdfter.find_vp_densitydifference(140)
# pdfter.find_vp_response(21, guess=True, svd_rcond=10**svdc, beta=0.1, a_rho_var=1e-7)
# pdfter.find_vp_cost_1basis(21, a_rho_var=1e-5, mu=1e-7)
hess, jac = pdfter.find_vp_response_1basis(35,
                                           ortho_basis=Orthogonal_basis,
                                           beta_method="Density",
                                           svd_rcond=svdc,
                                           regul_const=regulc,
                                           a_rho_var=1e-5,
                                           mu=1e-7
                                           )
# pdfter.find_vp_scipy_1basis(maxiter=140, opt_method=scipy_method, ortho_basis=Orthogonal_basis)

pdfter.vp_grid = vp_tester.to_grid(pdfter.vp[0])
f,ax = plt.subplots(1, 1, dpi=210)
ax.set_ylim(-1, 1)
pdft.plot1d_x(pdfter.vp_grid, vp_tester.Vpot, ax=ax,
              label="vp", color='black',
              dimmer_length=separation,
              title="vp")
# pdft.plot1d_x(pdfter.vp_Hext_nad, mol.Vpot, dimmer_length=separation,
#               title="vp" + title + str(pdfter.drho_conv[-1]), ax=ax, label="Hext", ls='--')
# pdft.plot1d_x(pdfter.vp_xc_nad, mol.Vpot, ax=ax, label="xc", ls='--')
# pdft.plot1d_x(pdfter.vp_kin_nad, mol.Vpot, ax=ax, label="kin", ls='--')
ax.legend()
f.show()
# f.savefig("TestSVDandRegu")
plt.close(f)

#%% 1 basis 2D plot
L = [3.0, 3.0, 2.0]
D = [0.1, 0.1, 0.1]
# Plot file
O, N = libcubeprop.build_grid(vp_tester.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(vp_tester.wfn, O, N, D)
if Orthogonal_basis:
    vp_cube = libcubeprop.compute_density_1basis(vp_tester.wfn, O, N, D, npoints, points, nxyz, block, np.dot(vp_tester.A.np, pdfter.vp[0]))
else:
    vp_cube = libcubeprop.compute_density_1basis(vp_tester.wfn, O, N, D, npoints, points, nxyz, block,
                                                 pdfter.vp[0])
f, ax = plt.subplots(1, 1, dpi=160)
p = ax.imshow(vp_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
atoms = libcubeprop.get_atoms(vp_tester.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
f.colorbar(p, ax=ax)
f.show()
plt.close(f)

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

# dvp_grid = mol.to_grid(jacL
# _approx)
# dvp_grid1 = mol.to_grid(jacL)
# dvp_grid2 = mol.to_grid(jac)
# f,ax = plt.subplots(1, 1, dpi=210)
# pdft.plot1d_x(dvp_grid, mol.Vpot, ax=ax, label="app", dimmer_length=separation)
# pdft.plot1d_x(dvp_grid1, mol.Vpot, ax=ax, label="dn-int vp*Cai", ls='--')
# pdft.plot1d_x(dvp_grid2, mol.Vpot, ax=ax, label='dn', ls='--')
# ax.legend()
# f.show()
# plt.close(f)