import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 4.522
functional = 'svwn'
basis = 'cc-pvdz'

psi4.set_output_file("Be2")

Full_Molec =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be  0 1 0
@Be  0 -1 0
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be  0 1 0
@Be  0 -1 0
@Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@Be %f 0.0 0.00
@Be  0 1 0
@Be  0 -1 0
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
pdfter.find_vp_response2(28, svd_rcond=1e-7, regul_const=1e-3, beta=0.1)
# pdfter.find_vp_densitydifference_onbasis(28, 1)
vp_grid = mol.to_grid(pdfter.vp[0])
pdft.plot1d_x(vp_grid, mol.Vpot, title="Be2 svd: 1e-3 l: 1e-9" + basis + functional)

pdfter.ep_conv = np.array(pdfter.ep_conv)
plt.plot(np.log10(np.abs(pdfter.ep_conv[1:] - pdfter.ep_conv[:-1])), 'o')
plt.title("log dEp")
plt.show()