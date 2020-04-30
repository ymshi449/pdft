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
svdc = 1e-4
regulc = None
Orthogonal_basis = False
lag_tap = 1
scipy_method = "dogleg"
# title = "Be WuYang1b Yan Q[nf] v[nf] svdc%i reguc%i " %(svdc, reguc) + basis + functional
title = F"ortho_vp_basis svd {svdc} regu {regulc} " + basis + functional + " orth_basis: " + str(Orthogonal_basis)
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

# Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
pdfter.fragments_scf_1basis(1000)

vp = np.zeros(mol.nbf)
pdfter.vp = [vp, vp]
vp_fock = psi4.core.Matrix.from_array(np.zeros_like(mol.H.np))
pdfter.vp_fock = [vp_fock, vp_fock]

grad, grad_app = pdfter.check_gradient_constrainedoptimization()