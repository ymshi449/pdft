import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

separation = 2
functional = 'svwn'
basis = 'cc-pvdz'
title = "N2 vxc inversion on "+basis
print(title)

psi4.set_output_file("N2.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
N %f 0.0 0.00
N -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))


Full_Molec.set_name("N2")

#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UKS'
})
E, input_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)

inverser = XC_Inversion.Inverser(mol, input_wfn)


inverser.find_vxc_scipy()

f,ax = plt.subplots(1,1,dpi=200)
XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_wfn.V_potential(), ax=ax,
                           dimmer_length=separation, label="input_xc_a", title=title)
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, mol.Vpot, ax=ax, label="WuYang_xc_a", ls='--')
ax.legend()
ax.set_xlim(-14,14)
ax.set_ylim(-7,0.1)
f.show()
plt.close(f)

# grad, grad_app = inverser.check_gradient()