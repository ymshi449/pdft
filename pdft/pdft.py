"""
pdft.py
"""


import psi4
import qcelemental as qc
import numpy as np


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
        self.frag_energy    = None
        self.energetics     = None
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
    def __init__(self, geometry, basis, method, omega=[None, None], mints=None, jk=None):
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
        self.ndocc      = self.wfn.nalpha()

        self.nalpha     = self.wfn.nalpha()
        self.nbeta      = self.wfn.nbeta()

        #From methods
        self.jk             = jk if jk is not None else self.form_JK()
        self.S              = self.mints.ao_overlap()
        self.A              = self.form_A()
        self.H              = self.form_H()

        #From SCF calculation
        self.Da             = None
        self.Db             = None
        self.energy         = None
        self.frag_energy    = None
        self.energetics     = None
        self.eig_a          = None
        self.eig_b          = None
        self.vks_a          = None
        self.vks_b          = None
        self.Fa             = None
        self.Fb             = None

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
            vp_a = psi4.core.Matrix(self.nbf,self.nbf)
            vp_b = psi4.core.Matrix(self.nbf,self.nbf)

            self.initialize()

        if vp_add == True:
            vp_a = vp_matrix[0]
            vp_b = vp_matrix[1]

        C_a, Cocc_a, D_a, eigs_a = build_orbitals(self.H, self.A, self.nalpha)
        C_b, Cocc_b, D_b, eigs_b = build_orbitals(self.H, self.A, self.nbeta)

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
            Vks_a.axpy(0.5, self.jk.J()[0])
            Vks_a.axpy(0.5, self.jk.J()[1])
            Vks_a.axpy(1.0, Vxc_a)

            Vks_b = self.mints.ao_potential()
            Vks_b.axpy(0.5, self.jk.J()[0])
            Vks_b.axpy(0.5, self.jk.J()[1])
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
            Partition = 1.0 * vp_a.vector_dot(D_a) + vp_b.vector_dot(D_b)
            Exchange_Correlation = ks_e

            SCF_E = Core
            SCF_E += (Hartree_a + Hartree_b) * 0.5
            SCF_E += Partition
            SCF_E += Exchange_Correlation            
            SCF_E += self.Enuc

            #print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
            #       % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))

            dRMS = 0.5 * (np.mean(diisa_e.np**2)**0.5 + np.mean(diisb_e.np**2)**0.5)

            print(F'SCF Convergence: dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')
            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
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
            print(F'\n')        
            print('Energy Contributions: ')
            print('\n')
            print(F'Core:                  {Core}')
            print(F'Hartree:              {(Hartree_a + Hartree_b) * 0.5}')
            print(F'Exchange Correlation:  {ks_e}')
            print(F'Partition Energy:      {Partition}')
            print(F'Nuclear Repulsion:     {self.Enuc}')
            print(F'Total Energy           {SCF_E}')
            print(F'\n')

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

        return

class U_Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments = fragments
        self.nfragments = len(fragments)
        self.molecule = molecule  # The entire system.

        #from mehtods
        self.fragments_Da = None
        self.fragments_Db = None

    def get_energies(self):
        total = []
        for i in range(len(self.fragments)):
            total.append(self.fragments[i].energies)
        total.append(self.molecule.energies)
        pandas = pd.concat(total,axis=1)
        return pandas

    def get_density_sum(self):
        sum_a = self.fragments[0].Da.np.copy()
        sum_b = self.fragments[0].Db.np.copy()

        for i in range(1,len(self.fragments)):
            sum_a +=  self.fragments[i].Da.np

        for i in range(1,len(self.fragments)):
            sum_b +=  self.fragments[i].Db.np
        self.fragments_Da = sum_a
        self.fragments_Db = sum_b

    def find_vp_response(self, maxiter=21, beta=None, atol=2e-4, guess=None):
        """
        Using the inverse of static response function to update dvp from a dn.
        See Jonathan's Thesis 5.4 5.5 5.6.
        :param maxiter: maximum iterations
        :param atol: convergence criteria
        :param guess: initial guess
        :return:
        """
        print("<<<<<<<<<<<<<<<<<<<<<<Compute_Method_Response<<<<<<<<<<<<<<<<<<<")
        # Prepare for tha auxiliary basis set.
        aux_basis = psi4.core.BasisSet.build(self.molecule.geometry, "DF_BASIS_SCF", "",
                                             "JKFIT", self.molecule.basis)
        S_Pmn_ao = np.squeeze(self.molecule.mints.ao_3coverlap(aux_basis, self.molecule.wfn.basisset(),
                                                               self.molecule.wfn.basisset()))
        S_Pmn_ao = 0.5*(np.transpose(S_Pmn_ao, (0, 2, 1)) + S_Pmn_ao)
        S_PQ = np.array(self.molecule.mints.ao_overlap(aux_basis, aux_basis))
        S_PQ = 0.5*(S_PQ.T + S_PQ)
        # S_Pm_ao = np.array(self.mints.ao_overlap(aux_basis, self.e_wfn.basisset()))
        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-15)
        S_PQinv = 0.5 * (S_PQinv.T + S_PQinv)
        fouroverlap = np.einsum('Pmn,PQ,Qrs->mnrs', S_Pmn_ao, S_PQinv, S_Pmn_ao, optimize=True)

        old_rho_conv = np.inf  # rho convergence (2 norm) of last iteration
        lamdb_lastupdate_iter = 0
        rho_convergance = []
        Ep_convergence = []
        _,_,_,w = self.molecule.Vpot.get_np_xyzw()

        if guess is None:
            vp_a = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Da.np))
            vp_b = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp = [vp_a, vp_b]
        # else:
        #    vp_guess

        if beta is None:
            beta = 1.0
        for scf_step in range(1, maxiter + 1):
            """
            For each fragment, v_p(r) = \sum_{alpha}C_{ij}dD_{mn}\phi_i(r)\phi_j(r)(ijmn) = C_{ij}dD_{mn}\phi_i(r)\phi_j(r)(Cij)(CD)^{-1}(Dmn)
            v_{p,uv} = \sum_{alpha}C_{ij}dD_{mn}(Aij)(AB)^{-1}(Buv)(Cij)(CD)^{-1}(Dmn)

            1) Un-orthogonalized
            2) I did not use alpha and beta wave functions to update Kai inverse. I should.
            """
            #   Update rho and change beta
            self.get_density_sum()
            rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
            rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
            # Based on the naive hope, whenever the current lamdb does not improve the density, get a smaller one.
            if old_rho_conv < np.sum(np.abs(rho_fragment - rho_molecule)*w):
                beta *= 0.7
                lamdb_lastupdate_iter = scf_step
            # If some lamdb has beed updating for a more than a long period, try to increase it to converge faster.
            elif (scf_step - lamdb_lastupdate_iter) > 21:
                beta /= 0.8
                lamdb_lastupdate_iter = scf_step

            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule)*w)
            rho_convergance.append(old_rho_conv)

            ## vp calculation
            # Store \sum_{alpha}C_{ij}
            C_a = np.zeros_like(S_Pmn_ao)
            C_b = np.zeros_like(S_Pmn_ao)
            for i in self.fragments:
                # GET dvp
                # matrices for epsilon_i - epsilon_j. M
                epsilon_occ_a = i.epsilon_a.np[:i.nalpha, None]
                epsilon_occ_b = i.epsilon_b.np[:i.nbeta, None]
                epsilon_unocc_a = i.epsilon_a.np[i.nalpha:]
                epsilon_unocc_b = i.epsilon_b.np[i.nbeta:]
                epsilon_a = epsilon_occ_a - epsilon_unocc_a
                epsilon_b = epsilon_occ_b - epsilon_unocc_b

                # S_Pmn_mo
                S_Pmn_mo_a = np.einsum('mi,nj,Pmn->Pij', i.Ca.np, i.Ca.np, S_Pmn_ao, optimize=True)
                S_Pmn_mo_b = np.einsum('mi,nj,Pmn->Pij', i.Cb.np, i.Cb.np, S_Pmn_ao, optimize=True)

                # Normalization
                fouroverlap_a = np.einsum('mij,nij,mn->ij', S_Pmn_mo_a[:, :i.wfn.nalpha(), i.wfn.nalpha():], S_Pmn_mo_a[:, :i.wfn.nalpha(), i.wfn.nalpha():], S_PQinv, optimize=True)
                fouroverlap_b = np.einsum('mij,nij,mn->ij', S_Pmn_mo_b[:, :i.wfn.nbeta(), i.wfn.nbeta():], S_Pmn_mo_b[:, :i.wfn.nbeta(), i.wfn.nbeta():], S_PQinv, optimize=True)
                fouroverlap_a += 1e-7
                fouroverlap_b += 1e-7
                C_a += np.einsum('ai,bj,Cij,ij -> Cab', i.Ca.np[:, :i.wfn.nalpha()], i.Ca.np[:, i.wfn.nalpha():], S_Pmn_mo_a[:, :i.wfn.nalpha(), i.wfn.nalpha():],
                                 epsilon_a/fouroverlap_a/(2*np.sqrt(2/np.pi)), optimize=True)
                C_b += np.einsum('ai,bj,Cij,ij -> Cab', i.Cb.np[:, :i.wfn.nbeta()], i.Cb.np[:, i.wfn.nbeta():], S_Pmn_mo_b[:, :i.wfn.nbeta(), i.wfn.nbeta():],
                                 epsilon_b/fouroverlap_b/(2*np.sqrt(2/np.pi)), optimize=True)

            # vp(r) = C_{Cab}(CD)^{-1}(Dmn)dD_(mn)\phi_a(r)\phi_b(r) = dvp_a/b_r_{ab}\phi_a(r)\phi_b(r)
            # Basically this is the coefficients of vp(r) on rhorho
            DaDiff = np.copy(self.fragments_Da - self.molecule.Da.np)
            DbDiff = np.copy(self.fragments_Db - self.molecule.Db.np)
            # vp(r) = C_{Cab}(CD)^{-1}(Dmn)dD_(mn)\phi_a(r)\phi_b(r) = dvp_a/b_r_{ab}\phi_a(r)\phi_b(r)
            delta_vp_a = np.einsum('Cab,CD,Dmn,mn -> ab', C_a, S_PQinv, S_Pmn_ao, -beta*DaDiff, optimize=True)
            delta_vp_b = np.einsum('Cab,CD,Dmn,mn -> ab', C_b, S_PQinv, S_Pmn_ao, -beta*DbDiff, optimize=True)

            delta_vp_a = 0.5*(delta_vp_a.T + delta_vp_a)
            delta_vp_b = 0.5*(delta_vp_b.T + delta_vp_b)

            delta_vp_a = np.einsum('ijmn,mn->ij', fouroverlap, delta_vp_a)
            delta_vp_b = np.einsum('ijmn,mn->ij', fouroverlap, delta_vp_b)

            delta_vp_a = psi4.core.Matrix.from_array(delta_vp_a)
            delta_vp_b = psi4.core.Matrix.from_array(delta_vp_b)

            vp_a.axpy(1.0, delta_vp_a)
            vp_b.axpy(1.0, delta_vp_b)
            vp_total.axpy(1.0, vp_a)
            vp_total.axpy(1.0, vp_b)
            vp = [vp_a, vp_b]
            # Update fragments info with vp we just git
            Ef = 0.0

            # Check for convergence
            for i in range(self.nfragments):
                self.fragments[i].scf(vp_add=True, vp_matrix=vp, maxiter=1400)
                Ef += self.fragments[i].energy
            Ef = Ef
            Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)
            # if np.isclose( total_densities.sum(),self.molecule.D.sum(), atol=1e-5) :
            if np.isclose(Ef, self.molecule.energy, atol):
                print("I WILL BREAK!")
                break
            elif beta < 1e-10:
                print("Break because even small step length can not improve.")
                break
            elif scf_step == maxiter:

                # raise Exception("Maximum number of SCF cycles exceeded for vp.")
                print("Maximum number of SCF cycles exceeded for vp.")

            print(
                F'Iteration: {scf_step-1} lamdb = {beta} Delta_D = {self.fragments_Da.sum() + self.fragments_Db.sum() - (self.molecule.Da.np.sum() + self.molecule.Db.np.sum())} Delta_Rho = {old_rho_conv}')
        return vp_a, vp_b, vp_total, rho_convergance, Ep_convergence

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
        _,_,_,w = self.molecule.Vpot.get_np_xyzw()
        print("+++++++++++++++++++++++++++++++HERE++++++++++++++++++++++++++++")
        if guess is None:
            vp_a = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Da.np))
            vp_b = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp_total = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.Db.np))
            vp =  [ vp_a , vp_b ]
        #else:
        #    vp_guess

        for scf_step in range(maxiter+1):

            total_density_a = np.zeros_like(self.molecule.Da.np)
            total_density_b = np.zeros_like(self.molecule.Db.np)
            total_energies = 0.0
            density_convergence = 0.0

            for i in range(self.nfragments):

                self.fragments[i].scf(vp_add=True, vp_matrix=vp)

                total_density_a += self.fragments[i].Da.np 
                total_density_b += self.fragments[i].Db.np 

                total_energies  += self.fragments[i].frag_energy

            #if np.isclose( total_densities.sum(),self.molecule.D.sum(), atol=1e-5) :
            if np.isclose(total_energies, self.molecule.energy, atol):
                break

        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        beta_lastupdate_iter = 0
        rho_convergence = []
        Ep_convergence = []
        for scf_step in range(1,maxiter+1):
            #if scf_step == maxiter:
            #    raise Exception("Maximum number of SCF cycles exceeded for vp.")
            self.get_density_sum()
            Ef = 0.0
            ## Tracking rho and changing beta
            rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
            rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
            # Based on the naive hope, whenever the current lamdb does not improve the density, get a smaller one.
            # if old_rho_conv < np.linalg.norm(rho_fragment - rho_molecule, ord=1):
            if old_rho_conv < np.sum(np.abs(rho_fragment - rho_molecule)*w):
                beta *= 0.9
                beta_lastupdate_iter = scf_step
            # If some lamdb has beed updating for a more than a long period, try to increase it to converge faster.
            elif (scf_step - beta_lastupdate_iter) > 3:
                beta /= 0.95
                beta_lastupdate_iter = scf_step
            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule)*w)
            rho_convergence.append(old_rho_conv)
            print(F'Iteration: {scf_step-1} Beta = {beta}   Delta_D = {self.fragments_Da.sum() + self.fragments_Db.sum() - (self.molecule.Da.np.sum() + self.molecule.Db.np.sum())} Delta_Rho = {old_rho_conv}')
            delta_vp_a = beta * (self.fragments_Da - self.molecule.Da.np)
            delta_vp_b = beta * (self.fragments_Db - self.molecule.Db.np)
            delta_vp_a = 0.5 * (delta_vp_a + delta_vp_a.T)
            delta_vp_b = 0.5 * (delta_vp_b + delta_vp_b.T)
            delta_vp_a =  psi4.core.Matrix.from_array( np.einsum('ijmn,mn->ij', S, delta_vp_a))
            delta_vp_b =  psi4.core.Matrix.from_array( np.einsum('ijmn,mn->ij', S, delta_vp_b))

            delta_vp_a = psi4.core.Matrix.from_array(delta_vp_a)
            delta_vp_b = psi4.core.Matrix.from_array(delta_vp_b)
            #
            vp_a.axpy(1.0, delta_vp_a)
            vp_b.axpy(1.0, delta_vp_b)
            #
            vp_total.axpy(1.0, vp_a)
            vp_total.axpy(1.0, vp_b)

        return vp_a, vp_b, vp_total

class Embedding:
    def __init__(self, fragments, molecule):
        #basics
        self.fragments = fragments
        self.nfragments = len(fragments)
        self.molecule = molecule

        #from mehtods
        self.fragment_densities = self.get_density_sum()

    def get_energies(self):
        total = []
        for i in range(len(self.fragments)):
            total.append(self.fragments[i].energies)
        total.append(self.molecule.energies)
        pandas = pd.concat(total,axis=1)
        return pandas

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
