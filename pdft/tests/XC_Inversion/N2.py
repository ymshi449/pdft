import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

separation = 5.122
functional = 'svwn'
basis = 'aug-pcsseg-2'
# basis = 'cc-pvqz'
ortho_basis = False
svd = 1e-5
opt_method="BFGS"
method = "WuYangMN"
title = "N2 "+ method + opt_method +" "+basis+" Ortho_basis: "\
        + str(ortho_basis) +" svd " + str(svd)
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
E, input_density_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)

inverser = XC_Inversion.Inverser(mol, input_density_wfn, ortho_basis=ortho_basis)

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_convergence()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    inverser.find_vxc_manualNewton(svd_rcond=svd)
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)

# dDa = input_density_wfn.Da().np - mol.Da.np
# dDb = input_density_wfn.Db().np - mol.Db.np
# dn = mol.to_grid(dDa + dDb)

f,ax = plt.subplots(1,1,dpi=200)
XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_density_wfn.V_potential(), ax=ax,
                           dimmer_length=separation, label="input_xc_a", title=title)
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, mol.Vpot, ax=ax, label="WuYang_xc_a", ls='--')
# XC_Inversion.pdft.plot1d_x(np.log10(np.abs(dn)), mol.Vpot, ax=ax, label="logdn", ls='dotted')
ax.legend()
ax.set_xlim(-14,14)
ax.set_ylim(-3,0.1)
f.show()
plt.close(f)