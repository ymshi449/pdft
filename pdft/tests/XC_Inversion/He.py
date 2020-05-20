import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

functional = 'svwn'
basis = 'sto-3g'
basis = 'cc-pvtz'
basis = 'aug-pcsseg-3'
basis = 'cc-pvdz'
basis = 'aug-cc-pvqz'

vp_basis = None

ortho_basis = False
svd = "search_segment_cycle"
opt_method="BFGS"
method = "WuYangScipy"
title = "Li2 "+ method +"/"+ opt_method + " " + basis+"/"+ \
        str(vp_basis) + " OB:"\
        + str(ortho_basis) + " svd:" + str(svd)
print(title)

psi4.set_output_file("He.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
He 0.0 0.0 0.0
units bohr
symmetry c1
""")


Full_Molec.set_name("He")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'REFERENCE' : 'UKS'
})
E, input_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vp_basis is not None:
    vp_basis = XC_Inversion.Molecule(Full_Molec, vp_basis, functional)
    vp_basis.scf_inversion(100)
else:
    vp_basis = mol

print("Number of Basis: ", mol.nbf, vp_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_wfn,
                                 ortho_basis=ortho_basis,
                                 vp_basis=vp_basis,
                                 v0="Hartree"
                                 )

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    # rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, back_tracking_method="LD")
    inverser.find_vxc_manualNewton(svd_rcond=svd, back_tracking_method="L",
                                   rho_conv_threshold=1e-10, svd_parameter=5)
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)

# dDa = input_wfn.Da().np - mol.Da.np
# dDb = input_wfn.Db().np - mol.Db.np
# dn = mol.to_grid(dDa + dDb)

inverser.get_vxc()
f,ax = plt.subplots(1,1,dpi=200)
XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_wfn.V_potential(), ax=ax,
                           label="input_xc_a", title=title)
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vp_basis.Vpot, ax=ax, label="WuYang_xc_a", ls='--')
# XC_Inversion.pdft.plot1d_x(np.log10(np.abs(dn)), mol.Vpot, ax=ax, label="logdn", ls='dotted')
ax.legend()
f.show()
plt.close(f)