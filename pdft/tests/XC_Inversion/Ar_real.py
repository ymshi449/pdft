import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(3)
    psi4.set_memory('4 GB')

functional = 'svwn'
basis = "cc-pcvdz"
vxc_basis = "cc-pcvqz"

ortho_basis = False
svd = "input_once"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"

title = method +"_"+ opt_method +"_"+v0+ "_" + basis+"_"+ \
        str(vxc_basis) + "_"\
        + str(ortho_basis) + "_" + str(svd)
print(title)

psi4.set_output_file("Ar.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Ar
units bohr
symmetry c1
""")

Full_Molec.set_name("Ar")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 302,
    # 'DFT_RADIAL_POINTS': 77,
    'MAXITER': 1000,
    # 'BASIS': basis,
    'REFERENCE': 'RHF'
})#  Get wfn for target density
_, input_density_wfn = psi4.gradient("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
# _, input_density_wfn = psi4.energy("SCF"+"/"+basis, molecule=Full_Molec, return_wfn=True)

print("Target Density Calculation Finished.")


#Psi4 Options:

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional)
    vxc_basis.scf(100)
else:
    vxc_basis = mol

print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0,
                                 # eHOMO=-0.5792,
                                 # v0_wfn=v0_wfn
                                 )
# grad, grad_app = inverser.check_gradient_WuYang()
# hess, hess_app = inverser.check_hess_WuYang()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B")

# f, ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vxc_basis.Vpot, ax=ax, label="WuYang", ls='--')
# ax.legend()
# ax.set_xlim(1e-3, 14)
# ax.set_xscale("log")
# f.show()