import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

functional = 'svwn'
basis = 'cc-pCVDZ'

vp_basis = 'cc-pCV5Z'

ortho_basis = False
svd = "segment_cycle_cutoff"
opt_method="trust-krylov"
method = "WuYangMN"
v0 = "FermiAmaldi"

title = method +"_"+ opt_method +"_"+v0+ "_" + basis+"_"+ \
        str(vp_basis) + "_"\
        + str(ortho_basis) + "_" + str(svd)
print(title)

psi4.set_output_file("FC2Cl.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
F -4.622780328 0 0
C -2.225662738 0 0
C 0        0 0
Cl 3.130803758 0 0
units bohr
symmetry c1
""")

Full_Molec.set_name("FC2Cl")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'REFERENCE' : 'UHF'
})
E, input_density_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)
#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vp_basis is not None:
    vp_basis = XC_Inversion.Molecule(Full_Molec, vp_basis, functional, jk="No Need for JK")
    print("Number of Basis: ", mol.nbf, vp_basis.nbf)
    # assert vp_basis.nbf < 230
    vp_basis.scf_inversion(10)
else:
    vp_basis = mol
    print("Number of Basis: ", mol.nbf, vp_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vp_basis,
                                 v0=v0
                                 )

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    # rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="LD")
    inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfeD")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)
#
# # dDa = input_density_wfn.Da().np - mol.Da.np
# # dDb = input_density_wfn.Db().np - mol.Db.np
# # dn = mol.to_grid(dDa + dDb)

# f,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_density_wfn.V_potential(), ax=ax, label="LDA")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vp_basis.Vpot, ax=ax, label="WuYang", ls='--')
# ax.legend()
# ax.set_xlim(-7.5,5.5)
# f.show()