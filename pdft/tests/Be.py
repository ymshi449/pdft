import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 4.522

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
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "cc-pvdz", "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
# vp_solution = pdfter.find_vp_optimizing(maxiter=29)
for svd in np.linspace(1, 7, 2):
    pdfter.find_vp_response2(21, svd_rcond=10**(-svd), regul_const=1e-5, beta=0.1)
    vp_grid = mol.to_grid(pdfter.vp[0])
    f, ax = plt.subplots(1,1,figsize=(16,12), dpi=160)
    pdft.plot1d_x(vp_grid, mol.Vpot, title="%.14f, Ep:% f, drho:% f"
                                           %(10**(-svd), pdfter.ep_conv[-1], pdfter.drho_conv[-1]),
                  dimmer_length=bondlength, ax=ax)
    f.show()
    f.savefig("VpBe2" + str(int(svd*100)))
    print("===============================================svd_rcond=%.14f, Ep:% f, drho:% f" %(10**(-svd), pdfter.ep_conv[-1], pdfter.drho_conv[-1]))