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
basis = 'cc-pvtz'
regulc = None
Orthogonal_basis = True
# title = "Be WuYang1b Yan Q[nf] v[nf] svdc%i reguc%i " %(svdc, reguc) + basis + functional
title = basis + functional + " orth_basis: " + str(Orthogonal_basis)
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

# Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
pdfter.fragments_scf_1basis(1000)

pdfter.CO_weight_exponent = 0.5

pdfter.find_vp_scipy_constrainedoptimization(100)
# pdfter.find_vp_constrainedoptimization_BT(770)
# pdft.plot1d_x(pdfter.vp_grid, mol.Vpot)

f,ax = plt.subplots(1, 1, dpi=210)
ax.set_ylim(-1, 1)
pdft.plot1d_x(pdfter.vp_grid, mol.Vpot, ax=ax,
              label="vp", color='black',
              dimmer_length=separation,
              title="vp")
# pdft.plot1d_x(pdfter.vp_Hext_nad, mol.Vpot, dimmer_length=separation,
#               title="vp" + title + str(pdfter.drho_conv[-1]), ax=ax, label="Hext", ls='--')
# pdft.plot1d_x(pdfter.vp_xc_nad, mol.Vpot, ax=ax, label="xc", ls='--')
# pdft.plot1d_x(pdfter.vp_kin_nad, mol.Vpot, ax=ax, label="kin", ls='--')
ax.legend()
f.show()

L = [3.0, 3.0, 2.0]
D = [0.1, 0.1, 0.1]
# Plot file
O, N = libcubeprop.build_grid(mol.wfn, L, D)
block, points, nxyz, npoints = libcubeprop.populate_grid(mol.wfn, O, N, D)
if Orthogonal_basis:
    vp_cube = libcubeprop.compute_density_1basis(mol.wfn, O, N, D, npoints, points, nxyz, block, np.dot(mol.A.np, pdfter.vp[0]))
else:
    vp_cube = libcubeprop.compute_density_1basis(mol.wfn, O, N, D, npoints, points, nxyz, block,
                                                 pdfter.vp[0])
f, ax = plt.subplots(1, 1, dpi=160)
p = ax.imshow(vp_cube[:, :, 20], interpolation="bicubic", cmap="Spectral")
atoms = libcubeprop.get_atoms(mol.wfn, D, O)
# ax.scatter(atoms[:,2], atoms[:,1])
f.colorbar(p, ax=ax)
f.show()
plt.close()