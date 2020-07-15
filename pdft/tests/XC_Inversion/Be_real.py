import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(3)
    psi4.set_memory('4 GB')

functional = 'svwn'
basis = 'cc-pcvdz'
vxc_basis = 'cc-pcvqz'

ortho_basis = False
svd = "input_once"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"

title = method +"_"+ opt_method +"_"+v0+ "_" + basis+"_"+ \
        str(vxc_basis) + "_"\
        + str(ortho_basis) + "_" + str(svd)
print(title)

psi4.set_output_file("Be.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Be
units bohr
symmetry c1
""")

Full_Molec.set_name("Be")

# Exact
Be = np.genfromtxt('/home/yuming/PDFT/pdft/pdft/data/Atom0/be.new8/Data')
Be_xyz = np.concatenate((-np.flip(Be[:, 1]), Be[:, 1]))
Be_vxc = np.concatenate((np.flip(Be[:, 3]), Be[:, 3]))

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'MAXITER': 1000,
    'BASIS': basis,
    'REFERENCE': 'RHF'
})
#  Get wfn for target density
E_input, input_density_wfn = psi4.energy("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
print("Target Density Calculation Finished.")


#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional)
    vxc_basis.scf_inversion(100)
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
# rgl_list, L_list, dT_list, P_list = inverser.my_L_curve_regularization();

# grad, grad_app = inverser.check_gradient_WuYang()
# hess, hess_app = inverser.check_hess_WuYang()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method, find_vxc_grid=False)
elif method == "WuYangMN":
    hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B")

# #
# f,ax = plt.subplots(1,1,dpi=200)
# ax.plot(Be_xyz, Be_vxc, label="Exact")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vxc_basis.Vpot, ax=ax, label="WuYang", ls='--')
# ax.legend()
# ax.set_xlim(1e-3, 14)
# ax.set_xscale("log")
# f.show()