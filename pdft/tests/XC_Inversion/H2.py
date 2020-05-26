import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

separation = 1.415*2.0
functional = 'svwn'
basis = 'sto-3g'
basis = 'aug-pcsseg-3'
basis = 'cc-pvtz'
# basis = 'aug-cc-pvqz'

vp_basis = None

ortho_basis = False
svd = "find_optimal_w_bruteforce"
opt_method="BFGS"
method = "WuYangScipy"
title = "H2 "+ method + opt_method + " " + basis + " Ortho_basis: "\
        + str(ortho_basis) + " svd " + str(svd)
print(title)

psi4.set_output_file("H2.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
H %f 0.0 0.00
H -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))


Full_Molec.set_name("H2")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'REFERENCE' : 'RHF'
})
E, input_wfn = psi4.energy('ccsd'+"/"+basis, molecule=Full_Molec, return_wfn=True)

psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)
print("Number of Basis: ", mol.nbf)
if vp_basis is not None:
    vp_basis = XC_Inversion.Molecule(Full_Molec, vp_basis, functional)
    vp_basis.scf(100)
else:
    vp_basis = mol

inverser = XC_Inversion.Inverser(mol, input_wfn,
                                 ortho_basis=ortho_basis,
                                 vp_basis=vp_basis)

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="LD")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)

dDa = input_wfn.Da().np - mol.Da.np
dDb = input_wfn.Db().np - mol.Db.np
dn = mol.to_grid(dDa + dDb)

# f,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_wfn.V_potential(), ax=ax,
#                            dimmer_length=separation, label="input_xc_a", title=title)
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, mol.Vpot, ax=ax, label="WuYang_xc_a", ls='--')
# # XC_Inversion.pdft.plot1d_x(np.log10(np.abs(dn)), mol.Vpot, ax=ax, label="logdn", ls='dotted')
# ax.legend()
# ax.set_xlim(-14,14)
# ax.set_ylim(-3,0.1)
# f.show()
# plt.close(f)