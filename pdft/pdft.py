"""
pdft.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "5" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "7" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=6

import psi4
import qcelemental as qc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

psi4.set_num_threads(3)


def build_orbitals(diag, A, ndocc):
    """
    Diagonalizes matrix

    Parameters
    ----------
    diag: psi4.core.Matrix
        Fock matrix

    A: psi4.core.Matrix
        A = S^(1/2), Produces orthonormalized Fock matrix

    ndocc: integer
        Number of occupied orbitals 

    Returns
    -------
    C: psi4.core.Matrix
        Molecular orbitals coefficient matrix
    
    Cocc: psi4.core.Matrix
        Occupied molecular orbitals coefficient matrix

    D: psi4.core.Matrix
        One-particle density matrix
    
    eigs: psi4.core.Vector
        Eigenvectors of Fock matrix
    """
    Fp = psi4.core.triplet(A, diag, A, True, False, True)

    nbf = A.shape[0]
    Cp = psi4.core.Matrix(nbf, nbf)
    eigvecs = psi4.core.Vector(nbf)
    Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(A, Cp, False, False)

    Cocc = psi4.core.Matrix(nbf, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D, eigvecs

def fouroverlap(wfn,geometry,basis, mints):
        """
        Calculates four overlap integral with Density Fitting method.

        Parameters
        ----------
        wfn: psi4.core.Wavefunction
            Wavefunction object of molecule

        geometry: psi4.core.Molecule
            Geometry of molecule

        basis: str
            Basis set used to build auxiliary basis set

        Return
        ------
        S_densityfitting: numpy array
            Four overlap tensor
        """
        aux_basis = psi4.core.BasisSet.build(geometry, "DF_BASIS_SCF", "",
                                             "JKFIT", basis)
        S_Pmn = np.squeeze(mints.ao_3coverlap(aux_basis, wfn.basisset(),
                                              wfn.basisset()))
        S_PQ = np.array(mints.ao_overlap(aux_basis, aux_basis))
        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-12)
        d_mnQ = np.einsum('Pmn,PQ->mnQ',S_Pmn,S_PQinv)
        S_densityfitting = np.einsum('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn, optimize=True)
        return S_densityfitting, d_mnQ, S_Pmn, S_PQ


def xc(D, Vpot, functional='lda'):
    """
    Calculates the exchange correlation energy and exchange correlation
    potential to be added to the KS matrix

    Parameters
    ----------
    D: psi4.core.Matrix
        One-particle density matrix
    
    Vpot: psi4.core.VBase
        V potential 

    functional: str
        Exchange correlation functional. Currently only supports RKS LSDA 

    Returns
    -------

    e_xc: float
        Exchange correlation energy
    
    Varr: numpy array
        Vxc to be added to KS matrix
    """
    nbf = D.shape[0]
    Varr = np.zeros((nbf, nbf))
    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for l_block in range(Vpot.nblocks()):
        
        # Obtain general grid information
        l_grid = Vpot.get_block(l_block)
        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        l_npoints = l_w.shape[0]

        points_func.compute_points(l_grid)

        # Compute the functional itself
        ret = superfunc.compute_functional(points_func.point_values(), -1)
        
        e_xc += np.vdot(l_w, np.array(ret["V"])[:l_npoints])
        v_rho = np.array(ret["V_RHO_A"])[:l_npoints]
    
        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global())
        points_func.compute_points(l_grid)
        nfunctions = lpos.shape[0]
        
        # Integrate the LDA
        phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]

        # LDA
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho, l_w, phi, optimize=True)
        
        # Sum back to the correct place
        Varr[(lpos[:, None], lpos)] += 0.5*(Vtmp + Vtmp.T)

    return e_xc, Varr


def U_xc(D_a, D_b, Vpot, functional='lda'):
    """
    Calculates the exchange correlation energy and exchange correlation
    potential to be added to the KS matrix

    Parameters
    ----------
    D: psi4.core.Matrix
        One-particle density matrix
    
    Vpot: psi4.core.VBase
        V potential 

    functional: str
        Exchange correlation functional. Currently only supports RKS LSDA 

    Returns
    -------

    e_xc: float
        Exchange correlation energy
    
    Varr: numpy array
        Vxc to be added to KS matrix
    """
    nbf = D_a.shape[0]
    V_a = np.zeros((nbf, nbf))
    V_b = np.zeros((nbf, nbf))
    
    total_e = 0.0
    
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    e_xc = 0.0
    
    # First loop over the outer set of blocks
    for l_block in range(Vpot.nblocks()):
        
        # Obtain general grid information
        l_grid = Vpot.get_block(l_block)
        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        l_npoints = l_w.shape[0]

        points_func.compute_points(l_grid)

        # Compute the functional itself
        ret = superfunc.compute_functional(points_func.point_values(), -1)
        
        e_xc += np.vdot(l_w, np.array(ret["V"])[:l_npoints])
        v_rho_a = np.array(ret["V_RHO_A"])[:l_npoints]
        v_rho_b = np.array(ret["V_RHO_B"])[:l_npoints]
    
        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global())
        points_func.compute_points(l_grid)
        nfunctions = lpos.shape[0]
        
        # Integrate the LDA
        phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]

        # LDA
        Vtmp_a = np.einsum('pb,p,p,pa->ab', phi, v_rho_a, l_w, phi, optimize=True)
        Vtmp_b = np.einsum('pb,p,p,pa->ab', phi, v_rho_b, l_w, phi, optimize=True)
        
        # Sum back to the correct place
        V_a[(lpos[:, None], lpos)] += 0.5*(Vtmp_a + Vtmp_a.T)
        V_b[(lpos[:, None], lpos)] += 0.5*(Vtmp_b + Vtmp_b.T)

    return e_xc, V_a,  V_b

class Molecule():
    def __init__(self, geometry, basis, method, mints=None, jk=None, restricted=True):
        #basics
        self.geometry   = geometry
        self.basis      = basis
        self.method     = method
        self.restricted = restricted
        self.Enuc       = self.geometry.nuclear_repulsion_energy()

        #Psi4 objects
        self.wfn        = psi4.core.Wavefunction.build(self.geometry, self.basis)
        self.functional = psi4.driver.dft.build_superfunctional(method, restricted=self.restricted)[0]
        self.mints = mints if mints is not None else psi4.core.MintsHelper(self.wfn.basisset())

        if restricted == True:
            resctricted_label = "RV"
        elif restricted == False:
            restricted_label  = "UV"
        self.Vpot       = psi4.core.VBase.build(self.wfn.basisset(), self.functional, resctricted_label)

        #From psi4 objects
        self.nbf        = self.wfn.nso()
        self.ndocc      = self.wfn.nalpha()

        #From methods
        self.jk             = jk if jk is not None else self.form_JK()
        self.S              = self.mints.ao_overlap()
        self.A              = self.form_A()
        self.H              = self.form_H()

        #From SCF
        self.C              = None
        self.Cocc           = None
        self.D              = None
        self.energy         = None
        self.frag_energy    = None  # frag_energy is the energy w/o contribution of vp
        self.energetics     = None  # energy is the energy w/ contribution of vp, \int vp*n.
        self.eigs           = None
        self.vks            = None

    def initialize(self):
        """
        Initializes functional and V potential objects
        """
        #Functional
        self.functional.set_deriv(2)
        self.functional.allocate()

        #External Potential
        self.Vpot.initialize()


    def form_H(self):
        """
        Forms core matrix 
        H =  T + V
        """
        V = self.mints.ao_potential()
        T = self.mints.ao_kinetic()
        H = T.clone()
        H.add(V)

        return H

    def form_JK(self, K=False):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.wfn.basisset())
        jk.set_memory(int(1.25e8)) #1GB
        jk.set_do_K(K)
        jk.initialize()
        jk.print_header()
        return jk

    def form_A(self):
        """
        Constructs matrix A = S^(1/2) required to orthonormalize the Fock Matrix
        """
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        return A

    def get_plot(self):
        plot = qc.models.Molecule.from_data(self.geometry.save_string_xyz())
        return plot

    def scf(self, maxiter=30, vp_add=False, vp_matrix=None, print_energies=False):
        """
        Performs scf calculation to find energy and density

        Parameters
        ----------
        vp: Bool
            Introduces a non-zero vp matrix

        vp_matrix: psi4.core.Matrix
            Vp_matrix to be added to KS matrix

        Returns
        -------

        """
        if vp_add == False:
            vp = psi4.core.Matrix(self.nbf,self.nbf)
            self.initialize()

        if vp_add == True:
            vp = vp_matrix



        C, Cocc, D, eigs = build_orbitals(self.H, self.A, self.ndocc)

        diis_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")

        for SCF_ITER in range(maxiter+1):

            self.jk.C_left_add(Cocc)
            self.jk.compute()
            self.jk.C_clear()

            #Bring core matrix
            F = self.H.clone()

            #Exchange correlation energy/matrix
            self.Vpot.set_D([D])
            self.Vpot.properties()[0].set_pointers(D)
            ks_e ,Vxc = xc(D, self.Vpot)
            Vxc = psi4.core.Matrix.from_array(Vxc)

            #add components to matrix
            F.axpy(2.0, self.jk.J()[0])
            F.axpy(1.0, Vxc)
            F.axpy(1.0, vp)

            #DIIS
            diis_e = psi4.core.triplet(F, D, self.S, False, False, False)
            diis_e.subtract(psi4.core.triplet(self.S, D, F, False, False, False))
            diis_e = psi4.core.triplet(self.A, diis_e, self.A, False, False, False)
            diis_obj.add(F, diis_e)
            dRMS = diis_e.rms()

            SCF_E  = 2.0 * self.H.vector_dot(D)
            SCF_E += 2.0 * self.jk.J()[0].vector_dot(D)
            SCF_E += ks_e
            SCF_E += self.Enuc
            SCF_E += 2.0 * vp.vector_dot(D)

            #print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
            #       % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))

            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                break

            Eold = SCF_E

            #DIIS extrapolate
            F = diis_obj.extrapolate()

            #Diagonalize Fock matrix
            C, Cocc, D, eigs = build_orbitals(F, self.A, self.ndocc)

            #Testing
            Vks = self.mints.ao_potential()
            Vks.axpy(2.0, self.jk.J()[0])
            Vks.axpy(1.0, Vxc)
            #Testing


            if SCF_ITER == maxiter:
                raise Exception("Maximum number of SCF cycles exceeded.")

        energetics = {"Core": 2.0 * self.H.vector_dot(D), "Hartree": 2.0 * self.jk.J()[0].vector_dot(D), "Exchange-Correlation":ks_e, "Nuclear": self.Enuc, "Total": SCF_E }

        self.C              = C
        self.Cocc           = Cocc
        self.D              = D
        self.energy         = SCF_E
        self.frag_energy    = SCF_E - 2.0 * vp.vector_dot(D) 
        self.energetics     = energetics
        self.eigs           = eigs
        self.vks            = Vks

        return



class U_Molecule():
    def __init__(self, geometry, basis, method, omega=1, mints=None, jk=None):
        """
        :param geometry:
        :param basis:
        :param method:
        :param omega: default as [None, None], means that integer number of occupation.
                      The entire system should always been set as [None, None].
                      For fragments, set as [omegaup, omegadown].
                      omegaup = floor(nup) - nup; omegadown = floor(ndown) - ndown
                      E[nup,ndown] = (1-omegaup-omegadowm)E[]
        :param mints:
        :param jk:
        """
        #basics
        self.geometry   = geometry
        self.basis      = basis
        self.method     = method
        self.Enuc = self.geometry.nuclear_repulsion_energy()

        #Psi4 objects
        self.wfn        = psi4.core.Wavefunction.build(self.geometry, self.basis)
        self.functional = psi4.driver.dft.build_superfunctional(method, restricted=False)[0]
        self.mints = mints if mints is not None else psi4.core.MintsHelper(self.wfn.basisset())
        self.Vpot       = psi4.core.VBase.build(self.wfn.basisset(), self.functional, "UV")

        #From psi4 objects
        self.nbf        = self.wfn.nso()
        self.ndocc      = self.wfn.nalpha() + self.wfn.nbeta() # what is this?

        self.nalpha     = self.wfn.nalpha()
        self.nbeta      = self.wfn.nbeta()

        #Fractional Occupation
        self.omega = omega

        #From methods
        self.jk             = jk if jk is not None else self.form_JK()
        self.S              = self.mints.ao_overlap()
        self.A              = self.form_A()
        self.H              = self.form_H()

        #From SCF calculation
        self.Da             = None
        self.Db             = None
        self.energy         = None
        self.frag_energy    = None  # frag_energy is the energy w/o contribution of vp
        self.energetics     = None  # energy is the energy w/ contribution of vp, \int vp*n.
        self.eig_a          = None
        self.eig_b          = None
        self.vks_a          = None
        self.vks_b          = None
        self.Fa             = None
        self.Fb             = None
        self.Ca             = None
        self.Cb             = None

    def initialize(self):
        """
        Initializes functional and V potential objects
        """
        #Functional
        self.functional.set_deriv(2)
        self.functional.allocate()

        #External Potential
        self.Vpot.initialize()

    def form_H(self):
        """
        Forms core matrix 
        H =  T + V
        """
        V = self.mints.ao_potential()
        T = self.mints.ao_kinetic()
        H = T.clone()
        H.add(V)

        return H

    def form_JK(self, K=False):
        """
        Constructs a psi4 JK object from input basis
        """
        jk = psi4.core.JK.build(self.wfn.basisset())
        jk.set_memory(int(1.25e8)) #1GB
        jk.set_do_K(K)
        jk.initialize()
        jk.print_header()
        return jk

    def form_A(self):
        """
        Constructs matrix A = S^(1/2) required to orthonormalize the Fock Matrix
        """
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        return A

    def get_plot(self):
        plot = qc.models.Molecule.from_data(self.geometry.save_string_xyz())
        return plot

    def to_grid(self, Duv, Duv_b=None, vpot=None):
        """
        For any function on double ao basis: f(r) = Duv*phi_u(r)*phi_v(r), e.g. the density.
        If Duv_b is not None, it will take Duv + Duv_b.
        One should use the same wfn for all the fragments and the entire systems since different geometry will
        give different arrangement of xyzw.
        :return: The value of f(r) on grid points.
        """
        if vpot is None:
            vpot = self.Vpot
        points_func = vpot.properties()[0]
        f_grid = np.array([])
        # Loop over the blocks
        for b in range(vpot.nblocks()):
            # Obtain block information
            block = vpot.get_block(b)
            points_func.compute_points(block)
            npoints = block.npoints()
            lpos = np.array(block.functions_local_to_global())

            # Compute phi!
            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

            # Build a local slice of D
            if Duv_b is None:
                lD = Duv[(lpos[:, None], lpos)]
            else:
                lD = Duv[(lpos[:, None], lpos)] + Duv_b[(lpos[:, None], lpos)]

            # Copmute rho
            f_grid = np.append(f_grid, np.einsum('pm,mn,pn->p', phi, lD, phi))
        return f_grid

    def to_basis(self, value, w=None):
        """
        For any function on integration grid points, get the coefficients on the basis set.
        The solution is not unique.
        value: array of values on points
        One should use the same wfn for all the fragments and the entire systems since different geometry will
        give different arrangement of xyzw.
        w: how many points to use for fitting. Default as None: use them all. If w, ROUGHLY w*nbf. w should always be greater than 1.
        :return: The value of f(r) on grid points.
        """
        vpot = self.Vpot
        points_func = vpot.properties()[0]
        nbf = self.nbf
        if w is not None:
            assert w>1, "w has to be greater than 1 !"
            w = int(w*nbf) + 1
        else:
            w = value.shape[0]
        basis_grid_matrix = np.empty((0, nbf ** 2))
        for b in range(vpot.nblocks()):
            # Obtain block information
            block = vpot.get_block(b)
            points_func.compute_points(block)
            npoints = block.npoints()
            lpos = np.array(block.functions_local_to_global())
            # Compute phi!
            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            appended = np.zeros((npoints, nbf ** 2))
            for i in range(0, npoints):
                appendelements = np.zeros((1, nbf))
                appendelements[0, lpos] = phi[i, :]
                appended[i, :] = np.squeeze((appendelements.T.dot(appendelements)).reshape(nbf ** 2, 1))
            appended = appended.reshape(npoints, nbf ** 2)
            basis_grid_matrix = np.append(basis_grid_matrix, appended, axis=0)
            if basis_grid_matrix.shape[0] >= w:
                break
        Da = np.linalg.lstsq(basis_grid_matrix, value[:basis_grid_matrix.shape[0]], rcond=None)
        Da = Da[0].reshape(nbf, nbf)
        Da = 0.5 * (Da + Da.T)
        return Da

    def scf(self, maxiter=30, vp_matrix=None, print_energies=False):
        """
        Performs scf calculation to find energy and density
        Parameters
        ----------
        vp: Bool
            Introduces a non-zero vp matrix

        vp_matrix: psi4.core.Matrix
            Vp_matrix to be added to KS matrix

        Returns
        -------
        """

        # if vp_add == False:
        #     vp_a = psi4.core.Matrix(self.nbf,self.nbf)
        #     vp_b = psi4.core.Matrix(self.nbf,self.nbf)
        #
        #     self.initialize()
        #
        # if vp_add == True:
        #     vp_a = vp_matrix[0]
        #     vp_b = vp_matrix[1]

        if vp_matrix is not None:
            vp_a = vp_matrix[0]
            vp_b = vp_matrix[1]
        else:
            vp_a = psi4.core.Matrix(self.nbf,self.nbf)
            vp_b = psi4.core.Matrix(self.nbf,self.nbf)
            vp_a.np[:] = 0.0
            vp_b.np[:] = 0.0
            self.initialize()


        if self.Da is None:
            C_a, Cocc_a, D_a, eigs_a = build_orbitals(self.H, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = build_orbitals(self.H, self.A, self.nbeta)
        else: # Use the calculation from last vp as initial.
            nbf = self.A.shape[0]
            Cocc_a = psi4.core.Matrix(nbf, self.nalpha)
            Cocc_a.np[:] = self.Ca.np[:, :self.nalpha]
            Cocc_b = psi4.core.Matrix(nbf, self.nbeta)
            Cocc_b.np[:] = self.Cb.np[:, :self.nbeta]
            D_a = self.Da
            D_b = self.Db

        diisa_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 
        diisb_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest") 

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")

        for SCF_ITER in range(maxiter+1):
            self.jk.C_left_add(Cocc_a)
            self.jk.C_left_add(Cocc_b)
            self.jk.compute()
            self.jk.C_clear()

            #Bring core matrix
            F_a = self.H.clone()
            F_b = self.H.clone()

            #Exchange correlation energy/matrix
            self.Vpot.set_D([D_a,D_b])
            self.Vpot.properties()[0].set_pointers(D_a, D_b)

            ks_e ,Vxc_a, Vxc_b = U_xc(D_a, D_b, self.Vpot)
            Vxc_a = psi4.core.Matrix.from_array(Vxc_a)
            Vxc_b = psi4.core.Matrix.from_array(Vxc_b)

            F_a.axpy(1.0, self.jk.J()[0])
            F_a.axpy(1.0, self.jk.J()[1]) 
            F_b.axpy(1.0, self.jk.J()[0])
            F_b.axpy(1.0, self.jk.J()[1])
            F_a.axpy(1.0, Vxc_a)
            F_b.axpy(1.0, Vxc_b)
            F_a.axpy(1.0, vp_a)
            F_b.axpy(1.0, vp_b)

            Vks_a = self.mints.ao_potential()
            Vks_a.axpy(0.5, self.jk.J()[0])  # why there is a 0.5
            Vks_a.axpy(0.5, self.jk.J()[1])  # why there is a 0.5
            Vks_a.axpy(1.0, Vxc_a)

            Vks_b = self.mints.ao_potential()
            Vks_b.axpy(0.5, self.jk.J()[0])  # why there is a 0.5
            Vks_b.axpy(0.5, self.jk.J()[1])  # why there is a 0.5
            Vks_b.axpy(1.0, Vxc_b)
            
            #DIIS
            diisa_e = psi4.core.triplet(F_a, D_a, self.S, False, False, False)
            diisa_e.subtract(psi4.core.triplet(self.S, D_a, F_a, False, False, False))
            diisa_e = psi4.core.triplet(self.A, diisa_e, self.A, False, False, False)
            diisa_obj.add(F_a, diisa_e)

            diisb_e = psi4.core.triplet(F_b, D_b, self.S, False, False, False)
            diisb_e.subtract(psi4.core.triplet(self.S, D_b, F_b, False, False, False))
            diisb_e = psi4.core.triplet(self.A, diisb_e, self.A, False, False, False)
            diisb_obj.add(F_b, diisb_e)

            dRMSa = diisa_e.rms()
            dRMSb = diisb_e.rms()

            Core = 1.0 * self.H.vector_dot(D_a) + 1.0 * self.H.vector_dot(D_b)
            Hartree_a = 1.0 * self.jk.J()[0].vector_dot(D_a) + self.jk.J()[1].vector_dot(D_a)
            Hartree_b = 1.0 * self.jk.J()[0].vector_dot(D_b) + self.jk.J()[1].vector_dot(D_b)
            Partition = vp_a.vector_dot(D_a) + vp_b.vector_dot(D_b)
            Exchange_Correlation = ks_e

            SCF_E = Core
            SCF_E += (Hartree_a + Hartree_b) * 0.5
            SCF_E += Partition
            SCF_E += Exchange_Correlation            
            SCF_E += self.Enuc

            #print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
            #       % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))

            dRMS = 0.5 * (np.mean(diisa_e.np**2)**0.5 + np.mean(diisb_e.np**2)**0.5)

            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                if print_energies is True:
                    print(F'SCF Convergence: NUM_ITER = {SCF_ITER} dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')
                break

            Eold = SCF_E

            #DIIS extrapolate
            F_a = diisa_obj.extrapolate()
            F_b = diisb_obj.extrapolate()

            #Diagonalize Fock matrix
            C_a, Cocc_a, D_a, eigs_a = build_orbitals(F_a, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = build_orbitals(F_b, self.A, self.nbeta)

            if SCF_ITER == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded.")
                print("Maximum number of SCF cycles exceeded.")
                if print_energies is True:
                    print(F'SCF Convergence: NUM_ITER = {SCF_ITER} dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')


        # if print_energies is True:
        #     print(F'\n')
        #     print('Energy Contributions: ')
        #     print('\n')
        #     print(F'Core:                  {Core}')
        #     print(F'Hartree:              {(Hartree_a + Hartree_b) * 0.5}')
        #     print(F'Exchange Correlation:  {ks_e}')
        #     print(F'Partition Energy:      {Partition}')
        #     print(F'Nuclear Repulsion:     {self.Enuc}')
        #     print(F'Total Energy           {SCF_E}')
        #     print(F'\n')

        energetics = {"Core":Core, "Hartree":(Hartree_a+Hartree_b)*0.5, "Exchange_Correlation":ks_e, "Nuclear":self.Enuc, "Total Energy":SCF_E}

        self.Da             = D_a
        self.Db             = D_b
        self.energy         = SCF_E
        self.frag_energy    = SCF_E - Partition
        self.energetics     = energetics
        self.eig_a          = eigs_a
        self.eig_b          = eigs_b
        self.vks_a          = Vks_a
        self.vks_b          = Vks_b
        self.Fa             = F_a
        self.Fb             = F_b
        self.Ca             = C_a
        self.Cb             = C_b

        return

    def flip_spin(self):
        """
        Flip the spin of given molecule: D, eps, C, Vks, F, nalpha&nbeta
        """
        temp = self.eig_a
        self.eig_a = self.eig_b
        self.eig_b = temp

        temp = self.vks_a
        self.vks_a = self.vks_b
        self.vks_b = temp

        temp = self.Fa
        self.Fa = self.Fb
        self.Fb = temp

        temp = self.Ca
        self.Ca = self.Cb
        self.Cb = temp

        temp = self.nalpha
        self.nalpha = self.nbeta
        self.nbeta = temp

        return

class U_Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments = fragments
        self.nfragments = len(fragments)
        self.molecule = molecule  # The entire system.

        # from mehtods, np array
        self.fragments_Da = None
        self.fragments_Db = None

        # vp
        self.vp_fock = None
        self.vp      = None  # Real function on basis

        self.four_overlap = None
        self.three_overlap = None

    def get_density_sum(self):
        sum_a = self.fragments[0].Da.np.copy() * self.fragments[0].omega
        sum_b = self.fragments[0].Db.np.copy() * self.fragments[0].omega

        for i in range(1, len(self.fragments)):
            sum_a +=  self.fragments[i].Da.np * self.fragments[i].omega
            sum_b +=  self.fragments[i].Db.np * self.fragments[i].omega

        self.fragments_Da = sum_a
        self.fragments_Db = sum_b
        return
    
    def initial_run(self, max_iter):
        self.molecule.scf(maxiter=max_iter, print_energies=True)

        for i in range(self.nfragments):
            self.fragments[i].scf(maxiter=max_iter, print_energies=True)

        self.get_density_sum()
        return

    def fragments_scf(self, max_iter, vp=None, vp_fock=None, printflag=False):
        # Run the whole molecule SCF calculation if not calculated before.
        if self.molecule.Da is None:
            self.molecule.scf(maxiter=max_iter, print_energies=printflag)

        if vp is None and vp_fock is None:
            # No vp is given.
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag)

        elif vp is True and vp_fock is None:
            if self.four_overlap is None:
                self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                         self.molecule.basis, self.molecule.mints)
            vp_fock = np.einsum('ijmn,mn->ij', self.four_overlap, self.vp[0])
            vp_fock = psi4.core.Matrix.from_array(vp_fock)
            self.vp_fock = [vp_fock, vp_fock]
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif (vp is not None and vp is not True) and vp_fock is None:
            self.vp = vp
            if self.four_overlap is None:
                self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                         self.molecule.basis, self.molecule.mints)
            vp_fock = np.einsum('ijmn,mn->ij', self.four_overlap, self.vp[0])
            vp_fock = psi4.core.Matrix.from_array(vp_fock)
            self.vp_fock = [vp_fock, vp_fock]
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif vp is None and vp_fock is True:
            # Zero self.vp so self.vp_fock does not correspond to an old version.
            self.vp = None

            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)

        elif vp is True and vp_fock is True:
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif (vp is not None and vp is not True) and vp_fock is True:
            self.vp = vp
            self.vp_fock = [vp_fock, vp_fock]
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif vp is None and (vp_fock is not None and vp_fock is not True):
            # Zero self.vp so self.vp_fock does not correspond to an old version.
            self.vp = None

            self.vp_fock = vp_fock
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)

        elif vp is True and (vp_fock is not None and vp_fock is not True):
            self.vp_fock = vp_fock
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif (vp is not None and vp is not True) and (vp_fock is not None and vp_fock is not True):
            self.vp = vp
            self.vp_fock = vp_fock
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        else:
            assert False, "If statement should never get here."

        self.get_density_sum()
        return
        
    def find_vp(self, beta, maxiter=21, guess=None, atol=2e-4):
        """
        Given a target function, finds vp_matrix to be added to each fragment
        ks matrix to match full molecule energy/density

        Parameters
        ----------
        beta: positive float
            Coefficient for delta_n = beta * (sum_fragment_densities - molecule_density)
        guess: Initial vp. Default None. If True, using self.vp and self.vp_fock. Otherwise, using given [vpa, vpb].
        Returns
        -------
        vp: psi4.core.Matrix
            Vp to be added to fragment ks matrix

        """
        # vp initialize
        # self.fragments[1].flip_spin()
        self.molecule.scf(maxiter=1000, print_energies=True)
        S, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                           self.molecule.basis, self.molecule.mints)
        if guess is None:
            vp_a = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_b = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))

            vp_afock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_bfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        elif guess is True:
            vp_a = self.vp[0]
            vp_b = self.vp[1]
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_total.np[:] += vp_a.np + vp_b.np

            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        else:
            vp_a = guess[0]
            vp_b = guess[1]
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_total.np[:] += vp_a.np + vp_b.np
            self.vp = guess

            vp_afock = np.einsum('ijmn,mn->ij', S, vp_a)
            vp_bfock = np.einsum('ijmn,mn->ij', S, vp_b)
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            self.vp_fock = [vp_totalfock, vp_totalfock]
            flag_update_vpfock = True
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega

        _,_,_,w = self.molecule.Vpot.get_np_xyzw()

        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        beta_lastupdate_iter = 0
        rho_convergence = []
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        Ep_convergence = []
        Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)

        ## vp update start
        for scf_step in range(1,maxiter+1):
            self.get_density_sum()
            ## Tracking rho and changing beta
            rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
            # Based on a naive hope, whenever the current beta does not improve the density, get a smaller one.
            if old_rho_conv < np.sum(np.abs(rho_fragment - rho_molecule)*w):
                beta *= 0.7
                beta_lastupdate_iter = scf_step
            # If some beta has beed used for a more than a long period, try to increase it to converge faster.
            elif (scf_step - beta_lastupdate_iter) > 3:
                beta /= 0.8
                beta_lastupdate_iter = scf_step
            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule)*w)
            rho_convergence.append(old_rho_conv)

            print(F'Iter: {scf_step-1} beta: {beta} dD: {np.linalg.norm(self.fragments_Da + self.fragments_Db - (self.molecule.Da.np + self.molecule.Db.np), ord=1)} d_rho: {old_rho_conv} Ep: {Ep_convergence[-1]}')

            delta_vp_a = beta * (self.fragments_Da - self.molecule.Da.np)
            delta_vp_b = beta * (self.fragments_Db - self.molecule.Db.np)
            delta_vp_a = 0.5 * (delta_vp_a + delta_vp_a.T)
            delta_vp_b = 0.5 * (delta_vp_b + delta_vp_b.T)

            vp_a += delta_vp_a
            vp_b += delta_vp_b
            vp_total += delta_vp_a + delta_vp_b
            self.vp = [vp_total, vp_total]

            delta_vp_a = np.einsum('ijmn,mn->ij', S, delta_vp_a)
            delta_vp_b = np.einsum('ijmn,mn->ij', S, delta_vp_b)

            vp_afock.np[:] += delta_vp_a
            vp_bfock.np[:] += delta_vp_b
            vp_totalfock.np[:] += delta_vp_a + delta_vp_b
            self.vp_fock = [vp_totalfock, vp_totalfock] # Use total_vp instead of spin vp for calculation.

            Ef = 0.0
            for i in range(self.nfragments):
                self.fragments[i].scf(vp_matrix=self.vp_fock, maxiter=1000)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)
            if False: #np.isclose(Ep_convergence[-2], Ep_convergence[-1], atol=atol):
                print("Break because Ep does not update")
                break
            elif beta < 1e-10:
                print("Break because even small step length can not improve.")
                break
            elif scf_step == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded for vp.")
                print("Maximum number of SCF cycles exceeded for vp.")

        return rho_convergence, Ep_convergence

    def hess(self, vp_array):
        """
        To get the Hessian operator on the basis set xi_p = phi_i*phi_j as a matrix.
        :return: Hessian matrix as np.array self.molecule.nbf**2 x self.molecule.nbf**2
        """
        if self.four_overlap is None:
            self.four_overlap = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                            self.molecule.basis, self.molecule.mints)[0]

        vp = vp_array.reshape(self.molecule.nbf, self.molecule.nbf)
        # If the vp stored is not the same as the vp we got, re-run scp calculations and update vp.
        if not np.linalg.norm(vp - self.vp[0]) < 1e-7:
            # update vp and vp fock
            self.vp = [vp, vp]
            self.fragments_scf(1000, vp=True)

        hess = np.zeros((self.molecule.nbf**2, self.molecule.nbf**2))
        for i in self.fragments:
            # GET dvp
            # matrices for epsilon_i - epsilon_j. M
            epsilon_occ_a = i.eig_a.np[:i.nalpha, None]
            epsilon_occ_b = i.eig_b.np[:i.nbeta, None]
            epsilon_unocc_a = i.eig_a.np[i.nalpha:]
            epsilon_unocc_b = i.eig_b.np[i.nbeta:]
            epsilon_a = epsilon_occ_a - epsilon_unocc_a
            epsilon_b = epsilon_occ_b - epsilon_unocc_b
            hess += i.omega*np.einsum('ai,bj,ci,dj,ij,amnb,cuvd -> mnuv', i.Ca.np[:, :i.nalpha], i.Ca.np[:, i.nalpha:],
                                      i.Ca.np[:, :i.nalpha], i.Ca.np[:, i.nalpha:], np.reciprocal(epsilon_a),
                                      self.four_overlap, self.four_overlap, optimize=True).reshape(self.molecule.nbf**2, self.molecule.nbf**2)
            hess += i.omega*np.einsum('ai,bj,ci,dj,ij,amnb,cuvd -> mnuv', i.Cb.np[:, :i.nbeta], i.Cb.np[:, i.nbeta:],
                                      i.Cb.np[:, :i.nbeta], i.Cb.np[:, i.nbeta:], np.reciprocal(epsilon_b),
                                      self.four_overlap, self.four_overlap, optimize=True).reshape(self.molecule.nbf**2, self.molecule.nbf**2)
        assert np.linalg.norm(hess - hess.T) < 1e-3, "hess not symmetry"
        hess = 0.5 * (hess + hess.T)
        print("Response", np.linalg.norm(hess))
        return hess

    def jac(self, vp_array):
        """
        To get Jaccobi vector, which is the density difference on the basis set xi_p = phi_i*phi_j.
        a + b
        :return: Jac, If matrix=False (default), vector as np.array self.molecule.nbf**2.
        If matrix=True, return a matrix self.molecule.nbf x self.molecule.nbf

        """

        vp = vp_array.reshape(self.molecule.nbf, self.molecule.nbf)
        # If the vp stored is not the same as the vp we got, re-run scp calculations and update vp.
        if not np.linalg.norm(vp - self.vp[0]) < 1e-7:
            # update vp and vp fock
            self.vp = [vp, vp]
            self.fragments_scf(1000, vp=True)

        if self.four_overlap is None:
            self.four_overlap = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                            self.molecule.basis, self.molecule.mints)[0]

        self.get_density_sum()
        density_difference_a = self.fragments_Da - self.molecule.Da.np
        density_difference_b = self.fragments_Db - self.molecule.Db.np

        jac = np.einsum("u,ui->i", (density_difference_a + density_difference_b).reshape(self.molecule.nbf**2),
                        self.four_overlap.reshape(self.molecule.nbf**2, self.molecule.nbf**2), optimize=True)
        return jac

    def lagrange_mul(self, vp_array):
        """
        Return Lagrange Multipliers (G) value.
        :return: L
        """
        vp = vp_array.reshape(self.molecule.nbf, self.molecule.nbf)
        # If the vp stored is not the same as the vp we got, re-run scp calculations and update vp.
        if not np.linalg.norm(vp - self.vp[0]) < 1e-7:
            # update vp and vp fock
            self.vp = [vp, vp]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] = np.einsum('ijmn,mn->ij', self.four_overlap, vp)
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # re-run scp
            for i in range(self.nfragments):
                # print("Calcualte fragment %i with new vp" %i)
                self.fragments[i].scf(vp_matrix=self.vp_fock, maxiter=100, print_energies=False)
        L = 0
        Ef = 0.0
        for i in range(self.nfragments):
            # print("Calcualte fragment %i with new vp" %i)
            L += self.fragments[i].energy*self.fragments[i].omega
            Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        Ep = self.molecule.energy - self.molecule.Enuc - Ef
        print("L: ", L, "Ef: ", Ef, "Ep: ", Ep)
        return L

    def find_vp_optimizing(self, maxiter=21, guess=None, opt_method="Newton-CG"):
        """
        WU-YANG
        :param maxiter:
        :param atol:
        :param guess:
        :return:
        """
        # Initial run
        self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                 self.molecule.basis, self.molecule.mints)
        self.molecule.scf(maxiter=1000, print_energies=True)
        if guess is None:
            vp_total = np.zeros_like(self.molecule.H.np)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            
            # if note given, use the first density difference to be initial
            self.get_density_sum()
            vp_total = self.fragments_Da - self.molecule.Da + self.fragments_Db - self.molecule.Db
            self.vp = [vp_total, vp_total]
            vp_totalfock.np[:] = np.einsum('ijmn,mn->ij', self.four_overlap, vp_total)
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # And run the iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega

        elif guess is True:
            vp_a = self.vp[0]
            vp_b = self.vp[1]
            vp_total = (vp_a.np + vp_b.np) * 0.5

            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            # otherwise, use the given one
        else:
            vp_a = guess[0]
            vp_b = guess[1]
            vp_total = (vp_a.np + vp_b.np) * 0.5
            self.vp = guess

            vp_afock = np.einsum('ijmn,mn->ij', self.four_overlap, vp_a)
            vp_bfock = np.einsum('ijmn,mn->ij', self.four_overlap, vp_b)
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        opt = {
            "disp": True,
            "maxiter": maxiter
        }

        # optimize using cipy, default as Newton-CG.
        vp_array = minimize(self.lagrange_mul, vp_total.reshape(self.molecule.nbf**2),
                            jac=self.jac, hess=self.hess, method=opt_method, options=opt)
        return vp_array

    def find_vp_response2(self, maxiter=21, beta=None, atol=1e-7, guess=None):
        """
        Using the inverse of static response function to update dvp from a dn.
        This version did inversion on xi_q =  psi_i*psi_j where psi is mo.
        See Jonathan's Thesis 5.4 5.5 5.6.
        :param maxiter: maximum iterations
        :param atol: convergence criteria
        :param guess: initial guess. When guess is True, object will look for self stored vp as initial.
        :return:
        """
        if guess is None:
            self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                     self.molecule.basis, self.molecule.mints)
            self.molecule.scf(maxiter=1000, print_energies=True)

            vp_total = np.zeros_like(self.molecule.H.np)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        elif guess is True:

            vp_total = self.vp[0]

            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            # Skip running the first iteration! When guess is True, everything is expected to be stored in this obj.
            Ef = np.Inf

        else:
            self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                     self.molecule.basis, self.molecule.mints)
            self.molecule.scf(maxiter=1000, print_energies=True)

            vp_total = guess[0]
            self.vp = guess

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(np.einsum('ijmn,mn->ij', self.four_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()

        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        beta_lastupdate_iter = 0
        rho_convergence = []
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        Ep_convergence = []
        Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)

        if beta is None:
            beta = 1.0

        print("<<<<<<<<<<<<<<<<<<<<<<Compute_Method_Response Method 2<<<<<<<<<<<<<<<<<<<")
        for scf_step in range(1, maxiter + 1):
            """
            For each fragment, v_p(r) = \sum_{alpha}C_{ij}dD_{mn}\phi_i(r)\phi_j(r)(ijmn) = C_{ij}dD_{mn}\phi_i(r)\phi_j(r)(Cij)(CD)^{-1}(Dmn)
            v_{p,uv} = \sum_{alpha}C_{ij}dD_{mn}(Aij)(AB)^{-1}(Buv)(Cij)(CD)^{-1}(Dmn)

            1) Un-orthogonalized
            2) I did not use alpha and beta wave functions to update Kai inverse. I should.
            """
            #   Update rho and change beta
            self.get_density_sum()
            rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
            # Based on the naive hope, whenever the current lamdb does not improve the density, get a smaller one.
            # if old_rho_conv < np.sum(np.abs(rho_fragment - rho_molecule) * w):
            #     beta *= 0.9
            #     beta_lastupdate_iter = scf_step
            # # If some lamdb has beed updating for a more than a long period, try to increase it to converge faster.
            # elif (scf_step - beta_lastupdate_iter) > 3:
            #     beta /= 0.8
            #     beta_lastupdate_iter = scf_step

            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule) * w)
            rho_convergence.append(old_rho_conv)

            print(
                F'Iter: {scf_step - 1} beta: {beta} dD: {np.linalg.norm(self.fragments_Da + self.fragments_Db - (self.molecule.Da.np + self.molecule.Db.np), ord=1)} '
                F'd_rho: {old_rho_conv} Ep: {Ep_convergence[-1]}')

            hess = self.response(self.vp[0].reshape(self.molecule.nbf**2))
            jac = self.density_difference(self.vp[0].reshape(self.molecule.nbf**2))
            # Solve the linear system by lstsq, because of singularity.
            dvp = np.linalg.lstsq(hess, beta*jac, rcond=None)[0]
            print("Solved?", np.linalg.norm(np.dot(hess, dvp) - beta*jac))
            vp_change = np.linalg.norm(dvp, ord=1)
            print("Imporvement", vp_change)
            dvp = dvp.reshape(self.molecule.nbf, self.molecule.nbf)

            vp_total += dvp
            # print(vp_total)
            self.vp = [vp_total, vp_total]

            dvpf = np.einsum('ijmn,mn->ij', self.four_overlap, dvp)

            vp_totalfock.np[:] += dvpf
            self.vp_fock = [vp_totalfock, vp_totalfock]  # Use total_vp instead of spin vp for calculation.

            # Update fragments info with vp we just git
            Ef = 0.0
            # Check for convergence
            for i in range(self.nfragments):
                # print("Calcualte fragment %i with new vp" %i)
                self.fragments[i].scf(vp_matrix=self.vp_fock, maxiter=30000, print_energies=False)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)
            if vp_change < 1e-10: #np.isclose(Ep_convergence[-2], Ep_convergence[-1], atol=atol):
                print("Break because Ep does not update")
                break
            elif beta < 1e-7:
                print("Break because even small step length can not improve.")
                break
            elif scf_step == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded for vp.")
                print("Maximum number of SCF cycles exceeded for vp.")

        return dvp, jac, hess

    def find_vp_response(self, maxiter=21, beta=None, atol=1e-7, guess=None):
        """
        Using the inverse of static response function to update dvp from a dn.
        This version did inversion on xi_q =  psi_i*psi_j where psi is mo.
        See Jonathan's Thesis 5.4 5.5 5.6.
        :param maxiter: maximum iterations
        :param atol: convergence criteria
        :param guess: initial guess
        :return:
        """
        # self.fragments[1].flip_spin()
        self.molecule.scf(maxiter=1000, print_energies=True)
        # Prepare for tha auxiliary basis set.
        aux_basis = psi4.core.BasisSet.build(self.molecule.geometry, "DF_BASIS_SCF", "",
                                             "JKFIT", self.molecule.basis)
        S_Pmn_ao = np.squeeze(self.molecule.mints.ao_3coverlap(aux_basis,
                                                               self.molecule.wfn.basisset(),
                                                               self.molecule.wfn.basisset()))
        S_Pmn_ao = 0.5 * (np.transpose(S_Pmn_ao, (0, 2, 1)) + S_Pmn_ao)
        S_PQ = np.array(self.molecule.mints.ao_overlap(aux_basis, aux_basis))
        S_PQ = 0.5 * (S_PQ.T + S_PQ)
        # S_Pm_ao = np.array(self.mints.ao_overlap(aux_basis, self.e_wfn.basisset()))
        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-15)
        S_PQinv = 0.5 * (S_PQinv.T + S_PQinv)
        fouroverlap = np.einsum('Pmn,PQ,Qrs->mnrs', S_Pmn_ao, S_PQinv, S_Pmn_ao, optimize=True)

        if guess is None:
            vp_a = np.zeros_like(self.molecule.H.np)
            vp_b = np.zeros_like(self.molecule.H.np)
            vp_total = np.zeros_like(self.molecule.H.np)

            vp_afock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_bfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        elif guess is True:
            vp_a = self.vp[0]
            vp_b = self.vp[1]
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_total.np[:] += vp_a.np + vp_b.np

            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            # Skip running the first iteration! When guess is True, everything is expected to be stored in this obj.
            Ef = np.Inf

        else:
            vp_a = guess[0]
            vp_b = guess[1]
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_total.np[:] += vp_a.np + vp_b.np
            self.vp = guess

            vp_afock = np.einsum('ijmn,mn->ij', fouroverlap, vp_a)
            vp_bfock = np.einsum('ijmn,mn->ij', fouroverlap, vp_b)
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=True, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()

        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        beta_lastupdate_iter = 0
        rho_convergence = []
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        Ep_convergence = []
        Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)

        if beta is None:
            beta = 1.0

        print("<<<<<<<<<<<<<<<<<<<<<<Compute_Method_Response<<<<<<<<<<<<<<<<<<<")
        for scf_step in range(1, maxiter + 1):
            """
            For each fragment, v_p(r) = \sum_{alpha}C_{ij}dD_{mn}\phi_i(r)\phi_j(r)(ijmn) = C_{ij}dD_{mn}\phi_i(r)\phi_j(r)(Cij)(CD)^{-1}(Dmn)
            v_{p,uv} = \sum_{alpha}C_{ij}dD_{mn}(Aij)(AB)^{-1}(Buv)(Cij)(CD)^{-1}(Dmn)

            1) Un-orthogonalized
            2) I did not use alpha and beta wave functions to update Kai inverse. I should.
            """
            #   Update rho and change beta
            self.get_density_sum()
            rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
            # Based on the naive hope, whenever the current lamdb does not improve the density, get a smaller one.
            if old_rho_conv < np.sum(np.abs(rho_fragment - rho_molecule) * w):
                beta *= 0.7
                beta_lastupdate_iter = scf_step
            # If some lamdb has beed updating for a more than a long period, try to increase it to converge faster.
            elif (scf_step - beta_lastupdate_iter) > 3:
                beta /= 0.8
                beta_lastupdate_iter = scf_step

            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule) * w)
            rho_convergence.append(old_rho_conv)

            print(
                F'Iter: {scf_step - 1} beta: {beta} dD: {np.linalg.norm(self.fragments_Da + self.fragments_Db - (self.molecule.Da.np + self.molecule.Db.np), ord=1)} '
                F'd_rho: {old_rho_conv} Ep: {Ep_convergence[-1]}')

            ## vp calculation
            # Store \sum_{alpha}C_{ij}
            C_a = np.zeros_like(S_Pmn_ao)
            C_b = np.zeros_like(S_Pmn_ao)
            for i in self.fragments:
                # GET dvp
                # matrices for epsilon_i - epsilon_j. M
                epsilon_occ_a = i.eig_a.np[:i.nalpha, None]
                epsilon_occ_b = i.eig_b.np[:i.nbeta, None]
                epsilon_unocc_a = i.eig_a.np[i.nalpha:]
                epsilon_unocc_b = i.eig_b.np[i.nbeta:]
                epsilon_a = epsilon_occ_a - epsilon_unocc_a
                epsilon_b = epsilon_occ_b - epsilon_unocc_b

                # S_Pmn_mo
                S_Pmn_mo_a = np.einsum('mi,nj,Pmn->Pij', i.Ca.np, i.Ca.np, S_Pmn_ao, optimize=True)
                S_Pmn_mo_b = np.einsum('mi,nj,Pmn->Pij', i.Cb.np, i.Cb.np, S_Pmn_ao, optimize=True)

                # Normalization
                fouroverlap_a = np.einsum('mij,nij,mn->ij', S_Pmn_mo_a[:, :i.nalpha, i.nalpha:],
                                          S_Pmn_mo_a[:, :i.nalpha, i.nalpha:], S_PQinv, optimize=True)
                fouroverlap_b = np.einsum('mij,nij,mn->ij', S_Pmn_mo_b[:, :i.nbeta, i.nbeta:],
                                          S_Pmn_mo_b[:, :i.nbeta, i.nbeta:], S_PQinv, optimize=True)
                fouroverlap_a += 1e-17
                fouroverlap_b += 1e-17
                C_a += np.einsum('ai,bj,Cij,ij -> Cab', i.Ca.np[:, :i.nalpha], i.Ca.np[:, i.nalpha:],
                                 S_Pmn_mo_a[:, :i.nalpha, i.nalpha:],
                                 epsilon_a / np.sqrt(fouroverlap_a) / (2 * np.sqrt(2 / np.pi)), optimize=True)
                C_b += np.einsum('ai,bj,Cij,ij -> Cab', i.Cb.np[:, :i.nbeta], i.Cb.np[:, i.nbeta:],
                                 S_Pmn_mo_b[:, :i.nbeta, i.nbeta:],
                                 epsilon_b / np.sqrt(fouroverlap_b) / (2 * np.sqrt(2 / np.pi)), optimize=True)
                # temp = np.einsum('ai,bj,Cij,ij -> Cab', i.Ca.np[:, :i.nalpha], i.Ca.np[:, i.nalpha:],
                #                  S_Pmn_mo_a[:, :i.nalpha, i.nalpha:],
                #                  epsilon_a/np.sqrt(fouroverlap_a)/(2*np.sqrt(2/np.pi)), optimize=True)
                # print(np.linalg.norm(np.einsum('Cab, CD, Dmn, mn -> ab', temp, S_PQinv, S_Pmn_ao, i.Da, optimize=True)))
            # vp(r) = C_{Cab}(CD)^{-1}(Dmn)dD_(mn)\phi_a(r)\phi_b(r) = dvp_a/b_r_{ab}\phi_a(r)\phi_b(r)
            # Basically this is the coefficients of vp(r) on rhorho
            DaDiff = np.copy(self.fragments_Da - self.molecule.Da.np)
            DbDiff = np.copy(self.fragments_Db - self.molecule.Db.np)
            # print("NORM", np.linalg.norm(C_a), np.linalg.norm(C_b))
            # vp(r) = C_{Cab}(CD)^{-1}(Dmn)dD_(mn)\phi_a(r)\phi_b(r) = dvp_a/b_r_{ab}\phi_a(r)\phi_b(r)
            delta_vp_a = np.einsum('Cab,CD,Dmn,mn -> ab', C_a, S_PQinv, S_Pmn_ao, - beta * DaDiff, optimize=True)
            delta_vp_b = np.einsum('Cab,CD,Dmn,mn -> ab', C_b, S_PQinv, S_Pmn_ao, - beta * DbDiff, optimize=True)

            delta_vp_a = 0.5 * (delta_vp_a + delta_vp_a.T)
            delta_vp_b = 0.5 * (delta_vp_b + delta_vp_b.T)

            vp_a += delta_vp_a
            vp_b += delta_vp_b
            vp_total += delta_vp_a + delta_vp_b
            self.vp = [vp_total, vp_total]

            delta_vp_a = np.einsum('ijmn,mn->ij', fouroverlap, delta_vp_a)
            delta_vp_b = np.einsum('ijmn,mn->ij', fouroverlap, delta_vp_b)

            vp_afock.np[:] += delta_vp_a
            vp_bfock.np[:] += delta_vp_b
            vp_totalfock.np[:] += delta_vp_a + delta_vp_b
            self.vp_fock = [vp_totalfock, vp_totalfock]  # Use total_vp instead of spin vp for calculation.

            # Update fragments info with vp we just git
            Ef = 0.0
            # Check for convergence
            for i in range(self.nfragments):
                # print("Calcualte fragment %i with new vp" %i)
                self.fragments[i].scf(vp_matrix=self.vp_fock, maxiter=30000, print_energies=False)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)
            if False:  # np.isclose(Ep_convergence[-2], Ep_convergence[-1], atol=atol):
                print("Break because Ep does not update")
                break
            elif beta < 1e-10:
                print("Break because even small step length can not improve.")
                break
            elif scf_step == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded for vp.")
                print("Maximum number of SCF cycles exceeded for vp.")

        return rho_convergence, Ep_convergence

class Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments = fragments
        self.nfragments = len(fragments)
        self.molecule = molecule

        #from mehtods
        self.fragment_densities = self.get_density_sum()

    def get_density_sum(self):
        sum = self.fragments[0].D.np.copy()
        for i in range(1,len(self.fragments)):
            sum +=  self.fragments[i].D.np
        return sum

    def find_vp(self, beta, guess=None, maxiter=10, atol=2e-4):
        """
        Given a target function, finds vp_matrix to be added to each fragment
        ks matrix to match full molecule energy/density

        Parameters
        ----------
        beta: positive float
            Coefficient for delta_n = beta * (molecule_density  - sum_fragment_densities)

        Returns
        -------
        vp: psi4.core.Matrix
            Vp to be added to fragment ks matrix

        """
        if guess==None:
            vp =  psi4.core.Matrix.from_array(np.zeros_like(self.molecule.D.np))
        #else:
        #    vp_guess

        for scf_step in range(maxiter+1):

            total_densities = np.zeros_like(self.molecule.D.np)
            total_energies = 0.0
            density_convergence = 0.0

            for i in range(self.nfragments):

                self.fragments[i].scf(vp_add=True, vp_matrix=vp)
                
                total_densities += self.fragments[i].D.np 
                total_energies  += self.fragments[i].frag_energy

            #if np.isclose( total_densities.sum(),self.molecule.D.sum(), atol=1e-5) :
            if np.isclose(total_energies, self.molecule.energy, atol):
                break

            #if scf_step == maxiter:
            #    raise Exception("Maximum number of SCF cycles exceeded for vp.")

            print(F'Iteration: {scf_step} Delta_E = {total_energies - self.molecule.energy} Delta_D = {total_densities.sum() - self.molecule.D.np.sum()}')

            delta_vp =  beta * (total_densities - self.molecule.D)  
            #S, D_mnQ, S_pmn, Spq = fouroverlap(self.fragments[0].wfn, self.fragments[0].geometry, "STO-3G", self.fragments[0].mints)
            #S_2, d_2, S_pmn_2, Spq_2 = fouroverlap(self.fragments[1].wfn, self.fragments[1].geometry, "STO-3G")

            #delta_vp =  psi4.core.Matrix.from_array( np.einsum('ijmn,mn->ij', S, delta_vp))
            delta_vp = psi4.core.Matrix.from_array(delta_vp)

            vp.axpy(1.0, delta_vp)

        return vp

def plot1d_x(data, Vpot, dimmer_length=2.0, title=None, fignum= None):
    """
    Plot on x direction
    :param data: Any f(r) on grid
    """
    x, y, z, w = Vpot.get_np_xyzw()
    # filter to get points on z axis
    mask = np.isclose(abs(y), 0, atol=1E-11)
    mask2 = np.isclose(abs(z), 0, atol=1E-11)
    order = np.argsort(x[mask & mask2])
    if fignum is None:
        # f1 = plt.figure(num=None, figsize=(16, 12), dpi=160)        
        f1 = plt.figure()
        plt.plot(x[mask & mask2][order], data[mask & mask2][order])
    else:
        # f1 = plt.figure(num=fignum, figsize=(16, 12), dpi=160)        
        f1 = plt.figure()
        plt.plot(x[mask & mask2][order], data[mask & mask2][order])
    plt.axvline(x=dimmer_length/2.0)
    plt.axvline(x=-dimmer_length/2.0)
    plt.xlabel("x-axis")
    if title is not None:
        plt.ylabel(title)
        plt.title(title + " plot on the X axis")
    plt.show()
