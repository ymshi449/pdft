import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(3)
    psi4.set_memory('4 GB')
separation = 5.122

functional = 'svwn'
basis = 'sto-3g'
basis = 'cc-pvtz'
basis = 'aug-cc-pvqz'
basis = 'aug-cc-pvdz'
basis = 'aug-cc-pvtz'
basis = 'aug-cc-pvqz'
basis = 'pcsseg-3'
basis = 'cc-pCVDZ'

vxc_basis = 'cc-pCVQZ'

ortho_basis = False
svd = "segment_cycle_cutoff"
opt_method="L-BFGS-B"
method = "WuYangScipy"
v0 = "Hartree"
v0 = "FermiAmaldi"

title = method +"_"+ opt_method +"_"+v0+ "_" + basis+"_"+ \
        str(vxc_basis) + "_"\
        + str(ortho_basis) + "_" + str(svd)
print(title)

psi4.set_output_file("Ne.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Li %f 0.0 0.00
Li -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))


Full_Molec.set_name("Ne")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'MAXITER': 1000,
    'BASIS': basis,
    'REFERENCE': 'RHF'

})
#  Get wfn for target density
E_input, input_density_wfn = psi4.energy("SCF"+"/"+basis, molecule=Full_Molec, return_wfn=True)
#  Get wfn for v0 using HF
# E_v0, v0_wfn = psi4.energy("scf"+"/"+basis, molecule=Full_Molec, return_wfn=True)
#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional)
    vxc_basis.scf(100)
else:
    vxc_basis = mol

print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 input_E=E_input,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0,
                                 # v0_wfn=v0_wfn
                                 )

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    # rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="LD")
    hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)

# dDa = input_density_wfn.Da().np - mol.Da.np
# dDb = input_density_wfn.Db().np - mol.Db.np
# dn = mol.to_grid(dDa + dDb)

# f,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vxc_basis.Vpot, ax=ax, label="WuYang_xc_a", ls='--')
# # XC_Inversion.pdft.plot1d_x(np.log10(np.abs(dn)), mol.Vpot, ax=ax, label="logdn", ls='dotted')
# ax.legend()
# ax.set_xlim(0,14)
# f.show()
# plt.close(f)