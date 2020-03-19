"""
pDFT.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6

import psi4
import qcelemental as qc
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimizer

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

    vxc = []
    
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

        vxc.extend(0.5*(v_rho_a+v_rho_b))

        # Sum back to the correct place
        V_a[(lpos[:, None], lpos)] += 0.5*(Vtmp_a + Vtmp_a.T)
        V_b[(lpos[:, None], lpos)] += 0.5*(Vtmp_b + Vtmp_b.T)

    return e_xc, V_a, V_b, np.array(vxc)

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
        # replace UKS from mol wfn
        self.wfn = psi4.driver.proc.scf_wavefunction_factory(self.method, self.wfn, "UKS")
        self.functional = psi4.driver.dft.build_superfunctional(method, restricted=False)[0]

        self.mints = mints if mints is not None else psi4.core.MintsHelper(self.wfn.basisset())
        self.Vpot       = self.wfn.V_potential()

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
        self.H, self.T, self.V = self.form_H()

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

        # vp component calculator
        self.esp_calculator = None
        self.vxc = None

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
        return H, T, V

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

    def two_gradtwo_grid(self, vpot=None):
        """
        Find \int phi_j*phi_n*dot(grad(phi_i), grad(phi_m)) to (ijmn)
        :param vpot:
        :return: twogradtwo (ijmn)
        """
        if vpot is None:
            vpot = self.Vpot
        points_func = vpot.properties()[0]
        points_func.set_deriv(1)
        twogradtwo = np.zeros((self.nbf, self.nbf, self.nbf, self.nbf))
        # Loop over the blocks
        for b in range(vpot.nblocks()):
            # Obtain block information
            block = vpot.get_block(b)
            points_func.compute_points(block)
            npoints = block.npoints()
            lpos = np.array(block.functions_local_to_global())
            w = block.w()

            # Compute phi!
            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
            phi_x = np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
            phi_y = np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
            phi_z = np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]

            inner = np.einsum("pa,pb->pab", phi_x, phi_x, optimize=True)
            inner += np.einsum("pa,pb->pab", phi_y, phi_y, optimize=True)
            inner += np.einsum("pa,pb->pab", phi_z, phi_z, optimize=True)

            idx = np.ix_(lpos,lpos,lpos,lpos)

            twogradtwo[idx] += np.einsum("pim,pj,pn,p->ijmn", inner, phi, phi, w, optimize=True)
        return twogradtwo

    def update_wfn_info(self):
        """
        Update wfn.Da and wfn.Db from self.Da and self.Db.
        :return:
        """
        self.wfn.Da().np[:] = self.Da.np
        self.wfn.Db().np[:] = self.Db.np
        # self.wfn.Ca().np[:] = self.Ca.np
        # self.wfn.Cb().np[:] = self.Cb.np
        # self.wfn.epsilon_a().np[:] = self.eig_a.np
        # self.wfn.epsilon_b().np[:] = self.eig_b.np
        return

    def esp_on_grid(self, grid=None, Vpot=None):
        """
        For given Da Db, get the esp on grid. Right now it does not work well. If you check
        the potential on the grid, it is not smooth. The density info should be in self.wfn.
        :param grid: N*3 np.array. If None, calculate the grid.
        :param Vpot: for calculating grid. Will be ignored if grid is given.
        :return: esp
        """
        # Breaks in multi-threading
        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)

        self.update_wfn_info()

        if self.esp_calculator is None:
            self.esp_calculator = psi4.core.ESPPropCalc(self.wfn)
        # Grid
        if grid is None:
            if Vpot is None:
                x, y, z, _ = self.Vpot.get_np_xyzw()
                grid = np.array([x, y, z])
                grid = psi4.core.Matrix.from_array(grid.T)
                assert grid.shape[1] == 3, "Grid should be N*3 np.array"
            else:
                x, y, z, _ = Vpot.get_np_xyzw()
                grid = np.array([x, y, z])
                grid = psi4.core.Matrix.from_array(grid.T)
                assert grid.shape[1] == 3, "Grid should be N*3 np.array"
        else:
            assert grid.shape[1] == 3, "Grid should be N*3 np.array"
            grid = psi4.core.Matrix.from_array(grid)

        # Calculate esp
        esp = self.esp_calculator.compute_esp_over_grid_in_memory(grid)
        esp = -esp.np

        # Set threads back.
        psi4.set_num_threads(nthreads)

        return esp

    def vext_on_grid(self, grid=None, Vpot=None):
        """
        The value of v_ext(r) on grid points.
        :param grid: N*3 np.array. If None, calculate the grid.
        :param Vpot: for calculating grid. Will be ignored if grid is given.
        :return:
        """
        natom = self.wfn.molecule().natom()
        nuclear_xyz = self.wfn.molecule().full_geometry().np

        Z = np.zeros(natom)
        # index list of real atoms. To filter out ghosts.
        zidx = []
        for i in range(natom):
            Z[i] = self.wfn.molecule().Z(i)
            if Z[i] != 0:
                zidx.append(i)

        if grid is None:
            if Vpot is None:
                grid = np.array(self.Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
            else:
                grid = np.array(Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
        vext = np.zeros(grid.shape[0])
        # Go through all real atoms
        for i in range(len(zidx)):
            R = np.sqrt(np.sum((grid - nuclear_xyz[zidx[i], :])**2, axis=1))
            vext += Z[zidx[i]]/R
            vext[R < 1e-15] = 0

        vext = -vext
        return vext

    def grid_to_fock(self, f):
        """
        Fock matrix integral on the grid.
        :param f: function values on grid. np array
        :return: V: f_fock matrix
        """

        V = np.zeros_like(self.Da.np)
        points_func = self.Vpot.properties()[0]

        i = 0
        # Loop over the blocks
        for b in range(self.Vpot.nblocks()):
            # Obtain block information
            block = self.Vpot.get_block(b)
            points_func.compute_points(block)
            npoints = block.npoints()
            lpos = np.array(block.functions_local_to_global())

            # Obtain the grid weight
            w = np.array(block.w())

            # Compute phi!
            phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

            Vtmp = np.einsum('pb,p,p,pa->ab', phi, f[i:i+npoints], w, phi, optimize=True)

            # Add the temporary back to the larger array by indexing, ensure it is symmetric
            V[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)

            i += npoints
        assert i == f.shape[0], "Did not run through all the points. %i %i" %(i, f.shape[0])
        return V


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
        if Duv.ndim == 2:
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
        elif Duv.ndim==1:
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
                    lD = Duv[lpos]
                else:
                    lD = Duv[lpos] + Duv_b[lpos]

                # Copmute rho
                f_grid = np.append(f_grid, np.einsum('pm,m->p', phi, lD))
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

            ks_e ,Vxc_a, Vxc_b, self.vxc = U_xc(D_a, D_b, self.Vpot)
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
            Vks_a.axpy(1.0, vp_a)

            Vks_b = self.mints.ao_potential()
            Vks_b.axpy(0.5, self.jk.J()[0])  # why there is a 0.5
            Vks_b.axpy(0.5, self.jk.J()[1])  # why there is a 0.5
            Vks_b.axpy(1.0, Vxc_b)
            Vks_b.axpy(1.0, vp_b)

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

        energetics = {"Core": Core, "Hartree":(Hartree_a+Hartree_b)*0.5, "Exchange_Correlation": ks_e, "Nuclear": self.Enuc, "Total Energy":SCF_E}

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
        self.vp_grid = None

        self.four_overlap = None
        self.three_overlap = None
        self.twogradtwo = None

        # convergence
        self.drho_conv = []
        self.ep_conv = []
        self.lagrange = []
        self.ef_conv = []

        # Regularization Constant
        self.regul_const = 0.0

        # nad parts of vp with local-Q
        self.vp_ext_nad = None
        self.vp_Hext_nad = None
        self.vp_xc_nad = None
        self.vp_kin_nad = None

    def get_density_sum(self):
        sum_a = self.fragments[0].Da.np.copy() * self.fragments[0].omega
        sum_b = self.fragments[0].Db.np.copy() * self.fragments[0].omega

        for i in range(1, len(self.fragments)):

            sum_a += self.fragments[i].Da.np * self.fragments[i].omega
            sum_b += self.fragments[i].Db.np * self.fragments[i].omega

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
            self.molecule.scf(print_energies=printflag)

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
            # self.vp = None

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
            # self.vp = None

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

        Ef = 0.0
        for i in range(self.nfragments):
            # print("Calcualte fragment %i with new vp" %i)
            Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        Ep = self.molecule.energy - self.molecule.Enuc - Ef

        self.ep_conv.append(Ep)
        self.ef_conv.append(Ef)

        self.get_density_sum()
        return

    def fragments_scf_1basis(self, max_iter, vp=None, vp_fock=None, printflag=False):
        """vp is now on 1 basis: vp = \sum b_i phi_i. In this case, only 3-overlap needed."""
        # Run the whole molecule SCF calculation if not calculated before.
        if self.molecule.Da is None:
            self.molecule.scf(maxiter=max_iter, print_energies=printflag)

        if vp is None and vp_fock is None:
            # No vp is given.
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(print_energies=printflag)

        elif vp is True and vp_fock is None:
            if self.three_overlap is None:
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            vp_fock = np.einsum('ijm,m->ij', self.three_overlap, self.vp[0])
            vp_fock = psi4.core.Matrix.from_array(vp_fock)
            self.vp_fock = [vp_fock, vp_fock]
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif (vp is not None and vp is not True) and vp_fock is None:
            self.vp = vp
            if self.three_overlap is None:
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            vp_fock = np.einsum('ijm,m->ij', self.three_overlap, self.vp[0])
            vp_fock = psi4.core.Matrix.from_array(vp_fock)
            self.vp_fock = [vp_fock, vp_fock]
            # Run the scf
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=max_iter, print_energies=printflag, vp_matrix=self.vp_fock)
        elif vp is None and vp_fock is True:
            # Zero self.vp so self.vp_fock does not correspond to an old version.
            # self.vp = None

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

        Ef = 0.0
        for i in range(self.nfragments):
            # print("Calcualte fragment %i with new vp" %i)
            Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        Ep = self.molecule.energy - self.molecule.Enuc - Ef

        self.ep_conv.append(Ep)
        self.ef_conv.append(Ef)

        self.get_density_sum()
        return

    def local_Q_on_grid(self, n_deno="input"):
        """
        Function to get local Q * omege for each fragments.
        :param n_deno: density in the denominator if "input", using density of molecule. If "nf", use nf. If "i", return 1.
        :return:
        """
        if n_deno is "input":
            nf = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        elif n_deno is "nf":
            self.get_density_sum()
            nf = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
        elif n_deno is "i":
            return [1]*self.nfragments

        n_alpha = []
        for i in self.fragments:
            n_alpha.append(self.molecule.to_grid(i.Da.np, Duv_b=i.Db.np)
                           * i.omega)

        Q = []
        for i in range(len(self.fragments)):
            Qtemp = n_alpha[i]/nf
            # Qtemp[nf<1e-14] = 0
            Q.append(Qtemp)
        return Q

    def get_vp_xc_nad(self, Qtype='nf'):
        """
        Get the vp_xc_nad (or call it esp) for each fragment with local Q approximation on grid.
        Only works for the the same number of up and down electron.
        :return: vp_ext_nad
        """
        Q = self.local_Q_on_grid(n_deno=Qtype)

        if Qtype != 'input':
            self.get_density_sum()
            v_xc_f = U_xc(self.fragments_Da, self.fragments_Da, self.molecule.Vpot)[-1]
        else:
            v_xc_f = self.molecule.vxc

        # v_xc of all fragments
        vp_xc_nad = v_xc_f
        for j in range(self.nfragments):
            vp_xc_nad -= self.fragments[j].vxc*Q[j]
        self.vp_xc_nad = vp_xc_nad
        return

    def get_vp_Hext_nad(self, Qtype='nf'):
        """
        Get the vp_ext_nad + vp_H_nad (or call it esp) for each fragment with local Q approximation on grid.
        :return: vp_ext_nad
        """
        Q = self.local_Q_on_grid(n_deno=Qtype)

        # v_Hext of all fragments
        if Qtype != 'input':
            self.get_density_sum()
            temp_mol_Da = self.molecule.wfn.Da().np
            temp_mol_Db = self.molecule.wfn.Db().np
            self.molecule.wfn.Da().np[:] = self.fragments_Da
            self.molecule.wfn.Db().np[:] = self.fragments_Db
            v_Hext_f = self.molecule.esp_on_grid()
            self.molecule.wfn.Da().np[:] = temp_mol_Da
            self.molecule.wfn.Db().np[:] = temp_mol_Db
        else:
            v_Hext_f = self.molecule.esp_on_grid().np

        vp_hext_nad = v_Hext_f
        for j in range(self.nfragments):
            vp_hext_nad -= self.fragments[j].esp_on_grid(Vpot=self.molecule.Vpot)*Q[j]
        self.vp_Hext_nad = vp_hext_nad
        return

    def get_vp_ext_nad(self, Qtype='nf'):
        """
        Get the vp_ext_nad for each fragment with local Q approximation on grid.
        :return: vp_ext_nad
        """

        Q = self.local_Q_on_grid(n_deno=Qtype)

        # Entire system v_ext
        mol_vext = self.molecule.vext_on_grid()

        vp_ext_nad = np.zeros_like(Q[0])
        for j in range(self.nfragments):
            vp_ext_nad += (mol_vext - self.fragments[j].vext_on_grid(Vpot=self.molecule.Vpot))*Q[j]
        self.vp_ext_nad = vp_ext_nad
        return

    def get_vp_nad(self, Qtype='nf', vstype='nf'):
        """
        Get vp_kin_nap by -vs[n_mol] - sum_alpha (-vs[n_alpha]), and vp_Hext_nad and vp_xc_nad
        :param Qtype: Type of Q function.
        :param vstype: Type of vs[n_{vstype}].
        :return:
        """
        Q = self.local_Q_on_grid(n_deno=Qtype)

        # vxc[nf]
        if vstype != 'input':
            self.get_density_sum()
            temp_mol_Da = self.molecule.wfn.Da().np
            temp_mol_Db = self.molecule.wfn.Db().np
            self.molecule.wfn.Da().np[:] = self.fragments_Da
            self.molecule.wfn.Db().np[:] = self.fragments_Db
            v_Hext_f = self.molecule.esp_on_grid()
            self.molecule.wfn.Da().np[:] = temp_mol_Da
            self.molecule.wfn.Db().np[:] = temp_mol_Db
        else:
            v_Hext_f = self.molecule.esp_on_grid()
        # v_Hext[nf]
        if vstype != 'input':
            self.get_density_sum()
            v_xc_f = U_xc(self.fragments_Da, self.fragments_Da, self.molecule.Vpot)[-1]
        else:
            v_xc_f = self.molecule.vxc

        # v_kin v_xc v_Hext for all fragments
        vp_hext_nad = v_Hext_f
        vp_xc_nad = v_xc_f
        vp_kin_nad = -v_xc_f - v_Hext_f
        for j in range(self.nfragments):
            temp_v_alpha_Hext_Q = self.fragments[j].esp_on_grid(Vpot=self.molecule.Vpot)*Q[j]
            temp_v_alpha_xc_Q = self.fragments[j].vxc * Q[j]
            vp_hext_nad -= temp_v_alpha_Hext_Q
            vp_xc_nad -= temp_v_alpha_xc_Q
            vp_kin_nad += temp_v_alpha_Hext_Q + temp_v_alpha_xc_Q
        self.vp_Hext_nad = vp_hext_nad
        self.vp_xc_nad = vp_xc_nad
        if Qtype != 'input':
            self.vp_kin_nad = vp_kin_nad + np.sum(Q * self.vp_grid, axis=0)
        else:
            self.vp_kin_nad = vp_kin_nad + np.sum(Q * self.vp_grid, axis=0)
        return

    def get_oueis_retularized_vp_nad(self, dvp=None, Qtype='nf', vstype='nf', vp_Hext_decomposition=False):
        """
        Using the track introduced by Oueis and Wasserman 2018
        to try to regularize vp_kin_nad and vp_xc_nad.
        :param Qtype:
        vp_Hext_decomposition=False. If true, it means self.vp=vp - vp_Hext_nad.
        :return:
        """
        if dvp is None:
            self.vp_grid = self.molecule.to_grid(self.vp[0])
        else:
            if self.vp_grid is not None:
                self.vp_grid += self.molecule.to_grid(dvp)
            else:
                self.vp_grid = self.molecule.to_grid(dvp)

        self.get_vp_nad(Qtype=Qtype, vstype=vstype)

        if not vp_Hext_decomposition:
            vp_kin_nad = self.vp_grid - self.vp_xc_nad - self.vp_Hext_nad
            vp_xc_nad = self.vp_grid - self.vp_kin_nad - self.vp_Hext_nad
        else:
            vp_kin_nad = self.vp_grid - self.vp_xc_nad
            vp_xc_nad = self.vp_grid - self.vp_kin_nad

        self.vp_kin_nad = vp_kin_nad
        self.vp_xc_nad = vp_xc_nad
        return

    def update_oueis_retularized_vp_nad(self, dvp=None, vp_Hext_decomposition=False, Qtype='nf', vstype='nf'):
        """
        Using regulized vp to update vp and vp_fock.
        Using the track introduced by Oueis and Wasserman 2018
        to try to regularize vp_kin_nad and vp_xc_nad.
        :return:
        """
        self.get_oueis_retularized_vp_nad(dvp=dvp, vp_Hext_decomposition=vp_Hext_decomposition, Qtype=Qtype, vstype=vstype)
        self.vp_grid = self.vp_kin_nad + self.vp_xc_nad + self.vp_Hext_nad
        vp_fock = psi4.core.Matrix.from_array(self.molecule.grid_to_fock(self.vp_grid))
        self.vp_fock = [vp_fock, vp_fock]
        return


    def find_vp_densitydifference(self, maxiter, beta, guess=None, rho_std=1e-5, printflag=False):
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
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                     self.molecule.basis, self.molecule.mints)
        if guess is None:
            vp_total = np.zeros_like(self.molecule.H.np)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf(1000, vp=True, vp_fock=True)
        elif guess is True:
            vp_total = self.vp[0]
            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
        else:
            vp_total = guess[0]
            self.vp = guess
            vp_totalfock = psi4.core.Matrix.from_array(
                np.zeros_like(np.einsum('ijmn,mn->ij', self.four_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf(1000, vp=True, vp_fock=True)

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()

        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        beta_lastupdate_iter = 0
        rho_convergence = []
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)

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

            print(F'Iter: {scf_step-1} beta: {beta} Ef: {self.ef_conv[-1]} Ep: {self.ep_conv[-1]} d_rho: {old_rho_conv}')

            delta_vp_a = beta * (self.fragments_Da - self.molecule.Da.np)
            delta_vp_b = beta * (self.fragments_Db - self.molecule.Db.np)
            delta_vp_a = 0.5 * (delta_vp_a + delta_vp_a.T)
            delta_vp_b = 0.5 * (delta_vp_b + delta_vp_b.T)

            vp_total += delta_vp_a + delta_vp_b
            self.vp = [vp_total, vp_total]

            delta_vp_a = np.einsum('ijmn,mn->ij', self.four_overlap, delta_vp_a)
            delta_vp_b = np.einsum('ijmn,mn->ij', self.four_overlap, delta_vp_b)

            vp_totalfock.np[:] += delta_vp_a + delta_vp_b
            self.vp_fock = [vp_totalfock, vp_totalfock] # Use total_vp instead of spin vp for calculation.

            self.fragments_scf(1000, vp=True, vp_fock=True)

            if beta < 1e-7:
                print("Break because even small step length can not improve.")
                break
            elif len(rho_convergence) >= 5:
                if np.std(rho_convergence[-4:]) < rho_std:
                    print("Break because rho does update for 5 iter")
                    break
            elif old_rho_conv < rho_std:
                print("Break because rho difference (cost) is small.")
                break
            # elif scf_step == maxiter:
            # raise Exception("Maximum number of SCF cycles exceeded for vp.")
            # print("Maximum number of SCF cycles exceeded for vp.")
        self.drho_conv = rho_convergence

        return

    def find_vp_densitydifference_onbasis(self, maxiter, beta, guess=None, rho_std=1e-5, printflag=False):
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

        self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                           self.molecule.basis, self.molecule.mints)
        Ep_convergence = []
        if guess is None:
            if self.four_overlap is None:
                self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                         self.molecule.basis, self.molecule.mints)
            self.molecule.scf(maxiter=1000, print_energies=printflag)

            vp_total = np.zeros_like(self.molecule.H.np)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=printflag)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)

            # # if guess not given, use the first density difference to be initial is probably a good idea.
            # Ef = 0.0
            # self.get_density_sum()
            # vp_total += beta*(self.fragments_Da - self.molecule.Da + self.fragments_Db - self.molecule.Db)
            # self.vp = [vp_total, vp_total]
            # vp_totalfock.np[:] += np.einsum('ijmn,mn->ij', self.four_overlap, vp_total)
            # self.vp_fock = [vp_totalfock, vp_totalfock]
            # # And run the iteration
            # for i in range(self.nfragments):
            #     self.fragments[i].scf(maxiter=1000, print_energies=printflag, vp_matrix=self.vp_fock)
            #     Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            # Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)
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

            vp_total = guess[0]
            self.vp = guess

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(np.einsum('ijmn,mn->ij', self.four_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=printflag, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()

        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        beta_lastupdate_iter = 0
        rho_convergence = []
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)

        ## vp update start
        print("<<<<<<<<<<<<<<<<<<<<<<Density Difference on Basis<<<<<<<<<<<<<<<<<<<")
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
            delta_vp_a = np.einsum('ijmn,mn->ij', self.four_overlap, delta_vp_a)
            delta_vp_b = np.einsum('ijmn,mn->ij', self.four_overlap, delta_vp_b)

            vp_total += delta_vp_a + delta_vp_b
            self.vp = [vp_total, vp_total]

            delta_vp_a = np.einsum('ijmn,mn->ij', self.four_overlap, delta_vp_a)
            delta_vp_b = np.einsum('ijmn,mn->ij', self.four_overlap, delta_vp_b)

            vp_totalfock.np[:] += delta_vp_a + delta_vp_b
            self.vp_fock = [vp_totalfock, vp_totalfock] # Use total_vp instead of spin vp for calculation.

            Ef = 0.0
            for i in range(self.nfragments):
                self.fragments[i].scf(vp_matrix=self.vp_fock, maxiter=1000)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
            Ep_convergence.append(self.molecule.energy - self.molecule.Enuc - Ef)
            if beta < 1e-7:
                print("Break because even small step length can not improve.")
                break
            elif len(rho_convergence) >= 5:
                if np.std(rho_convergence[-4:]) < rho_std:
                    print("Break because rho does update for 5 iter")
                    break
            elif old_rho_conv < rho_std:
                print("Break because rho difference (cost) is small.")
                break
            # elif scf_step == maxiter:
            # raise Exception("Maximum number of SCF cycles exceeded for vp.")
            # print("Maximum number of SCF cycles exceeded for vp.")
        self.drho_conv = rho_convergence
        self.ep_conv = Ep_convergence

        return

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
        # assert np.linalg.norm(hess - hess.T) < 1e-3, "hess not symmetry"
        hess = 0.5 * (hess + hess.T)

        # Regularization
        if self.regul_const is not None:
            T = self.twogradtwo.reshape(self.molecule.nbf**2, self.molecule.nbf**2)
            T = 0.5 * (T + T.T)
            hess -= 4*4*self.regul_const*T

        # print("Response", np.linalg.norm(hess))
        # print(hess)
        return -hess

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

        # Regularization
        if self.regul_const is not None:
            T = self.twogradtwo.reshape(self.molecule.nbf**2, self.molecule.nbf**2)
            T = 0.5 * (T + T.T)
            jac -= 4*4*self.regul_const*np.dot(T, vp_array)

        # print("Jac norm:", np.linalg.norm(jac))
        return -jac

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
            self.fragments_scf(1000, vp=True)

        density_difference_a = self.fragments_Da - self.molecule.Da.np
        density_difference_b = self.fragments_Db - self.molecule.Db.np

        Ef = self.ef_conv[-1]
        Ep = self.ep_conv[-1]

        L = Ef
        L += np.sum(self.vp_fock[0].np*(density_difference_a + density_difference_b))

        # Regularization
        if self.regul_const is not None:
            T = self.twogradtwo.reshape(self.molecule.nbf**2, self.molecule.nbf**2)
            T = 0.5 * (T + T.T)
            print(L, T.shape, vp.shape)
            L -= 4*4*self.regul_const*np.dot(np.dot(vp_array, T), vp_array)

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        self.get_density_sum()
        rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
        rho_conv = np.sum(np.abs(rho_fragment - rho_molecule) * w)

        self.drho_conv.append(rho_conv)
        self.lagrange.append(-L)

        print("-L:", -L, "Int_vp_drho:", L-Ef, "Ef:", Ef, "Ep: ", Ep, "drho:", rho_conv)
        return -L

    def find_vp_scipy(self, maxiter=21, guess=None, regul_const=None, opt_method="Newton-CG"):
        """
        Scipy Newton-CG
        :param maxiter:
        :param atol:
        :param guess:
        :return:
        """
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                     self.molecule.basis, self.molecule.mints)
        if guess is None:
            vp_total = np.zeros_like(self.molecule.H.np)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf(1000, vp=True, vp_fock=True)
        elif guess is True:
            vp_total = self.vp[0]
            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
        else:
            vp_total = guess[0]
            self.vp = guess
            vp_totalfock = psi4.core.Matrix.from_array(
                np.zeros_like(np.einsum('ijmn,mn->ij', self.four_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf(1000, vp=True, vp_fock=True)

        self.regul_const = regul_const
        if self.twogradtwo is None and self.regul_const is not None:
            self.twogradtwo = self.molecule.two_gradtwo_grid()

        opt = {
            "disp": True,
            "maxiter": maxiter,
            "eps": 1e-7
        }
        # optimize using cipy, default as Newton-CG.
        vp_array = optimizer.minimize(self.lagrange_mul, self.vp[0].reshape(self.molecule.nbf**2),
                                      jac=self.jac, hess=self.hess, method=opt_method, options=opt)
        return vp_array

    def find_vp_response(self, maxiter, beta=None, guess=None, beta_update=None,
                         vp_nad_iter=None,
                         svd_rcond=None, regul_const=None, a_rho_var=1e-4, vp_norm_conv=1e-6, printflag=False):
        """
        Using the inverse of static response function to update dvp from a dn.
        This version describe vp = sum b_ij*phi_i*phi_j. phi is ao.

        See Jonathan's Thesis 5.4 5.5 5.6. and WuYang's paper
        :param maxiter: maximum vp update iterations
        :param guess: initial guess. When guess is True, object will look for self stored vp as initial.
        :param beta: step length for Newton's method.
        :param vp_nad_component: True. If False, will not calculate vp_Hext_nad.
        :param vp_nad_iter: 1. The number of iterations vp_Hext will be updated
        :param svd_rcond np.lingal.pinv rcond for hess psudo-inverse
        :param regul_const regularization constant.
        :param a_rho_var convergence threshold for last 5 drho std
        :param vp_norm_conv convergence threshold vp coefficient norm
        :param printflag printing flag
        :return:
        """
        if self.four_overlap is None:
            self.four_overlap, _, _, _ = fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                     self.molecule.basis, self.molecule.mints)
        if guess is None:
            vp_total = np.zeros_like(self.molecule.H.np)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf(1000, vp=True, vp_fock=True)
        elif guess is True:
            vp_total = self.vp[0]
            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
        else:
            vp_total = guess[0]
            self.vp = guess
            vp_totalfock = psi4.core.Matrix.from_array(
                np.zeros_like(np.einsum('ijmn,mn->ij', self.four_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf(1000, vp=True, vp_fock=True)

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()
        ## Tracking rho and changing beta
        old_rho_conv = np.inf
        # beta_lastupdate_iter = 0
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)

        if vp_nad_iter is not None:
            self.get_density_sum()
            rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
            print("no vp drho:", np.sum(np.abs(rho_fragment - rho_molecule) * w))
            self.get_vp_Hext_nad()
            vp_Hext_nad_fock = self.molecule.grid_to_fock(self.vp_Hext_nad)
            vp_totalfock.np[:] += vp_Hext_nad_fock
            self.vp_fock = [vp_totalfock, vp_totalfock]
            self.fragments_scf(1000, vp_fock=True)

        ## Tracking rho and changing beta
        L_old = np.inf

        if beta is None:
            beta = 0.1

        self.regul_const = regul_const

        if svd_rcond is None:
            svd_rcond = 1e-3

        if self.twogradtwo is None and self.regul_const is not None:
            self.twogradtwo = self.molecule.two_gradtwo_grid()

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
            if beta_update is not None:
                L = self.lagrange_mul(self.vp[0].reshape(self.molecule.nbf**2))
                # # Based on the naive hope, whenever the current lamdb does not improve the density, get a smaller one.
                if L >= L_old:
                    # print("\n L_old - L %.14f \n" %(L - L_old))
                    beta *= beta_update
                L_old = L

            # Update vp_none_add from time to time

            if vp_nad_iter is not None:
                if scf_step%vp_nad_iter == 0:
                    vp_totalfock.np[:] -= vp_Hext_nad_fock
                    self.get_vp_Hext_nad()
                    vp_Hext_nad_fock = self.molecule.grid_to_fock(self.vp_Hext_nad)
                    vp_totalfock.np[:] += vp_Hext_nad_fock
                    self.vp_fock = [vp_totalfock, vp_totalfock]

            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule) * w)
            self.drho_conv.append(old_rho_conv)

            print(F'Iter: {scf_step - 1} beta: {beta}'
                  F'Ef: {self.ef_conv[-1]} Ep: {self.ep_conv[-1]} d_rho: {old_rho_conv}')

            hess = self.hess(self.vp[0].reshape(self.molecule.nbf**2))
            jac = self.jac(self.vp[0].reshape(self.molecule.nbf**2))

            # Solve by SVD
            hess_inv = np.linalg.pinv(hess, rcond=svd_rcond)
            dvp = hess_inv.dot(beta*jac)
            vp_change = np.linalg.norm(dvp, ord=1)
            if printflag:
                print("Solved?", np.linalg.norm(np.dot(hess, dvp) - beta*jac))
                print("dvp norm", vp_change)
            dvp = -dvp.reshape(self.molecule.nbf, self.molecule.nbf)
            dvp = 0.5 * (dvp + dvp.T)
            vp_total += dvp
            self.vp = [vp_total, vp_total]

            dvpf = np.einsum('ijmn,mn->ij', self.four_overlap, dvp)

            vp_totalfock.np[:] += dvpf
            self.vp_fock = [vp_totalfock, vp_totalfock]  # Use total_vp instead of spin vp for calculation.
            print(F'norm dvpfock: {np.linalg.norm(dvpf)}')

            self.fragments_scf(1000, vp_fock=True)

            if beta < 1e-7:
                print("Break because even small step length can not improve.")
                break
            elif len(self.drho_conv) >= 5:
                if np.std(self.drho_conv[-4:]) < a_rho_var and vp_change < vp_norm_conv:
                    print("Break because rho and vp do not update for 5 iterations.")
                    break
            elif old_rho_conv < 1e-4:
                print("Break because rho difference (cost) is small.")
                break
            # elif scf_step == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded for vp.")
                # print("Maximum number of SCF cycles exceeded for vp.")
        return

    def hess_1basis(self, vp):
        """
        To get the Hessian operator on the basis set xi_p = phi_i as a matrix.
        :return: Hessian matrix as np.array self.molecule.nbf**2 x self.molecule.nbf**2
        """
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

        # If the vp stored is not the same as the vp we got, re-run scp calculations and update vp.
        if not np.linalg.norm(vp - self.vp[0]) < 1e-7:
            # update vp and vp fock
            self.vp = [vp, vp]
            self.fragments_scf_1basis(1000, vp=True)

        hess = np.zeros((self.molecule.nbf, self.molecule.nbf))
        for i in self.fragments:
            # GET dvp
            # matrices for epsilon_i - epsilon_j. M
            epsilon_occ_a = i.eig_a.np[:i.nalpha, None]
            epsilon_occ_b = i.eig_b.np[:i.nbeta, None]
            epsilon_unocc_a = i.eig_a.np[i.nalpha:]
            epsilon_unocc_b = i.eig_b.np[i.nbeta:]
            epsilon_a = epsilon_occ_a - epsilon_unocc_a
            epsilon_b = epsilon_occ_b - epsilon_unocc_b
            hess += i.omega*np.einsum('ai,bj,ci,dj,ij,amb,cnd -> mn', i.Ca.np[:, :i.nalpha], i.Ca.np[:, i.nalpha:],
                                      i.Ca.np[:, :i.nalpha], i.Ca.np[:, i.nalpha:], np.reciprocal(epsilon_a),
                                      self.three_overlap, self.three_overlap, optimize=True)
            hess += i.omega*np.einsum('ai,bj,ci,dj,ij,amb,cnd -> mn', i.Cb.np[:, :i.nbeta], i.Cb.np[:, i.nbeta:],
                                      i.Cb.np[:, :i.nbeta], i.Cb.np[:, i.nbeta:], np.reciprocal(epsilon_b),
                                      self.three_overlap, self.three_overlap, optimize=True)
        # assert np.linalg.norm(hess - hess.T) < 1e-3, "hess not symmetry"
        hess = 0.5 * (hess + hess.T)

        # Regularization
        if self.regul_const is not None:
            T = self.molecule.T.np
            T = 0.5 * (T + T.T)
            hess -= 4*4*self.regul_const*T

        # print("Response", np.linalg.norm(hess))
        # print(hess)
        return -hess

    def jac_1basis(self, vp):
        """
        To get Jaccobi vector, which is the density difference on the basis set xi_p = phi_i.
        a + b
        :return: Jac, If matrix=False (default), vector as np.array self.molecule.nbf**2.
        If matrix=True, return a matrix self.molecule.nbf x self.molecule.nbf

        """
        # If the vp stored is not the same as the vp we got, re-run scp calculations and update vp.
        if not np.linalg.norm(vp - self.vp[0]) < 1e-7:
            # update vp and vp fock
            self.vp = [vp, vp]
            self.fragments_scf_1basis(1000, vp=True)

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

        self.get_density_sum()
        density_difference_a = self.fragments_Da - self.molecule.Da.np
        density_difference_b = self.fragments_Db - self.molecule.Db.np

        jac = np.einsum("uv,uiv->i", (density_difference_a + density_difference_b), self.three_overlap, optimize=True)

        # Regularization
        if self.regul_const is not None:
            T = self.molecule.T.np
            T = 0.5 * (T + T.T)
            jac -= 4*4*self.regul_const*np.dot(T, vp)

        # print("Jac norm:", np.linalg.norm(jac))
        return -jac

    def lagrange_mul_1basis(self, vp):
        """
        Return Lagrange Multipliers (G) value. on 1 basis.
        :return: L
        """
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

        # If the vp stored is not the same as the vp we got, re-run scp calculations and update vp.
        if not np.linalg.norm(vp - self.vp[0]) < 1e-7:
            # update vp and vp fock
            self.vp = [vp, vp]
            self.fragments_scf_1basis(1000, vp=True)

        Ef = self.ef_conv[-1]
        Ep = self.ep_conv[-1]

        self.get_density_sum()
        density_difference_a = self.fragments_Da - self.molecule.Da.np
        density_difference_b = self.fragments_Db - self.molecule.Db.np

        L = Ef
        L += np.sum(self.vp_fock[0].np*(density_difference_a + density_difference_b))


        # Regularization
        if self.regul_const is not None:
            T = self.molecule.T.np
            T = 0.5 * (T + T.T)
            L -= 4*4*self.regul_const*np.dot(np.dot(vp, T), vp)

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()
        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        self.get_density_sum()
        rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
        rho_conv = np.sum(np.abs(rho_fragment - rho_molecule) * w)

        self.drho_conv.append(rho_conv)
        # self.ep_conv.append(Ep)
        self.lagrange.append(-L)

        # print("-L:", -L, "Int_vp_drho:", L-Ef, "Ef:", Ef, "Ep: ", Ep, "drho:", rho_conv)
        return -L

    def find_vp_scipy_1basis(self, maxiter=21, guess=None, regul_const=None, opt_method="Newton-CG", printflag=False):
        """
        Scipy Newton-CG
        :param maxiter:
        :param atol:
        :param guess:
        :return:
        """
        self.regul_const = regul_const

        if guess is None:
            if self.three_overlap is None:
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            self.molecule.scf(maxiter=1000, print_energies=printflag)

            vp_total = np.zeros(self.molecule.nbf)
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=printflag)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        elif guess is True:

            vp_total = self.vp[0]

            vp_afock = self.vp_fock[0]
            vp_bfock = self.vp_fock[1]
            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            vp_totalfock.np[:] += vp_afock.np + vp_bfock.np
            # Skip running the first iteration! When guess is True, everything is expected to be stored in this obj.
        else:
            if self.three_overlap is None:
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            self.molecule.scf(maxiter=1000, print_energies=printflag)

            vp_total = guess[0]
            self.vp = guess

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(np.einsum('ijm,m->ij', self.three_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]
            # Initialize
            Ef = 0.0
            # Run the first iteration
            for i in range(self.nfragments):
                self.fragments[i].scf(maxiter=1000, print_energies=printflag, vp_matrix=self.vp_fock)
                Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang 1 basis Scipy<<<<<<<<<<<<<<<<<<<")
        opt = {
            "disp": True,
            "maxiter": maxiter,
            "eps": 1e-7
        }
        # optimize using cipy, default as Newton-CG.
        vp_array = optimizer.minimize(self.lagrange_mul_1basis, self.vp[0],
                                      jac=self.jac_1basis, hess=self.hess_1basis, method=opt_method, options=opt)
        return vp_array

    def find_vp_response_1basis(self, maxiter=21, guess=None, beta=None, vp_nad_iter=None, Qtype='nf', vstype='nf',
                                svd_rcond=None,
                                beta_update=None, regul_const=None, a_rho_var=1e-4,
                                vp_norm_conv=1e-6, printflag=False):
        """
        Using the inverse of static response function to update dvp from a dn.
        This version describe vp = sum b_i*phi_i. phi is ao.
        See Jonathan's Thesis 5.4 5.5 5.6. and WuYang's paper
        :param maxiter: maximum vp update iterations
        :param guess: initial guess. When guess is True, object will look for self stored vp as initial.
        :param beta: step length for Newton's method.
        :param vp_nad_iter: 1. The number of iterations vp_Hext will be updated. If None, will not use vp_nad components.
        :param svd_rcond np.lingal.pinv rcond for hess psudo-inverse
        :param beta_update: If not None, will update beta by beta *= beta_update when L increased compared to last iteration.
        :param regul_const regularization constant.
        :param a_rho_var convergence threshold for last 5 drho std
        :param vp_norm_conv convergence threshold vp coefficient norm
        :param printflag printing flag
        :return:
        """

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())

        if guess is None:
            vp_total = np.zeros(self.molecule.H.np.shape[0])
            self.vp = [vp_total, vp_total]

            vp_totalfock = psi4.core.Matrix.from_array(np.zeros_like(self.molecule.H.np))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf_1basis(1000, vp=True, vp_fock=True)
        elif guess is True:
            vp_total = self.vp[0]
            vp_totalfock = self.vp_fock[0]
        else:
            vp_total = guess[0]
            self.vp = guess
            vp_totalfock = psi4.core.Matrix.from_array(
                np.zeros_like(np.einsum('ijmn,mn->ij', self.four_overlap, guess[0])))
            self.vp_fock = [vp_totalfock, vp_totalfock]

            self.fragments_scf_1basis(1000, vp=True, vp_fock=True)

        self.vp_ext_nad = self.get_vp_ext_nad()
        vp_ext_nad_fock = self.molecule.grid_to_fock(self.vp_ext_nad)
        vp_totalfock.np[:] += vp_ext_nad_fock
        self.vp_fock = [vp_totalfock, vp_totalfock]
        # Initialize
        Ef = 0.0
        # Run the first iteration
        for i in range(self.nfragments):
            self.fragments[i].scf(maxiter=1000, print_energies=printflag)
            Ef += (self.fragments[i].frag_energy - self.fragments[i].Enuc) * self.fragments[i].omega
        self.ep_conv.append(self.molecule.energy - self.molecule.Enuc - Ef)

        _, _, _, w = self.molecule.Vpot.get_np_xyzw()

        ## Tracking rho and changing beta
        L_old = np.inf

        rho_molecule = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)

        # if vp_nad_iter is not None:
        #     self.get_density_sum()
        #     rho_fragment = self.molecule.to_grid(self.fragments_Da, Duv_b=self.fragments_Db)
        #     print("no vp drho:", np.sum(np.abs(rho_fragment - rho_molecule) * w))
        #     self.get_vp_Hext_nad()
        #     vp_Hext_nad_fock = self.molecule.grid_to_fock(self.vp_Hext_nad)
        #     vp_totalfock.np[:] += vp_Hext_nad_fock
        #     self.vp_fock = [vp_totalfock, vp_totalfock]
        #     self.fragments_scf_1basis(1000, vp_fock=True)

        if beta is None:
            beta = 1.0

        self.regul_const = regul_const

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang 1 basis manual Newton<<<<<<<<<<<<<<<<<<<")
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
            old_rho_conv = np.sum(np.abs(rho_fragment - rho_molecule) * w)
            self.drho_conv.append(old_rho_conv)

            if beta_update is not None:
                L = self.lagrange_mul_1basis(self.vp[0])
                # # Based on the naive hope, whenever the current lamdb does not improve the density, get a smaller one.
                if L >= L_old:
                    # print("\n L_old - L %.14f \n" %(L - L_old))
                    beta *= beta_update
                L_old = L
                if scf_step%10 == 0:
                    beta /= beta_update
                    if beta > 1:
                        beta = 1

            # Update vp_none_add from time to time
            if vp_nad_iter is not None:
                if scf_step%vp_nad_iter == 0:
                    vp_totalfock.np[:] -= vp_Hext_nad_fock
                    self.get_vp_Hext_nad()
                    vp_Hext_nad_fock = self.molecule.grid_to_fock(self.vp_Hext_nad)
                    vp_totalfock.np[:] += vp_Hext_nad_fock
                    self.vp_fock = [vp_totalfock, vp_totalfock]

            print(
                F'Iter: {scf_step - 1} beta: {beta}'
                F' Ef: {self.ef_conv[-1]} Ep: {self.ep_conv[-1]} d_rho: {old_rho_conv}')

            hess = self.hess_1basis(self.vp[0])
            jac = self.jac_1basis(self.vp[0])

            # Solve by SVD
            if svd_rcond is not None:
                hess_inv = np.linalg.pinv(hess, rcond=svd_rcond)
                dvp = -beta*hess_inv.dot(jac)
                vp_change = np.linalg.norm(dvp, ord=1)
                if printflag:
                    print("Solved?", np.linalg.norm(np.dot(hess, dvp) - beta*jac))
                    print("dvp norm", vp_change)
            else:
                vp_change_last = 1e7
                for i in np.linspace(1, 10, 20):
                    # Solve by SVD
                    hess_inv = np.linalg.pinv(hess, rcond=10 ** -i)
                    dvp_temp = -beta * hess_inv.dot(jac)
                    vp_change = np.linalg.norm(dvp_temp, ord=1)
                    if vp_change/vp_change_last > 10 and i > 1:
                        dvp = dvp_last
                        if printflag:
                            print("scdc", i)
                            print(vp_change_last, vp_change)
                            print("dvp norm", np.linalg.norm(np.dot(hess, dvp_temp) - beta * jac))
                        break
                    dvp_last = dvp_temp
                    vp_change_last = vp_change

            # Traditional WuYang
            vp_total += dvp
            self.vp = [vp_total, vp_total]
            dvpf = np.einsum('ijm,m->ij', self.three_overlap, dvp)
            dvpf = 0.5 * (dvpf + dvpf.T)
            vp_totalfock.np[:] += dvpf
            self.vp_fock = [vp_totalfock, vp_totalfock]  # Use total_vp instead of spin vp for calculation.

            # With Yan's trick
            # vp_total += dvp
            # self.vp = [vp_total, vp_total]
            # self.update_oueis_retularized_vp_nad(Qtype=Qtype, vstype=vstype,
            #                                      vp_Hext_decomposition=False if (vp_nad_iter is None) else True)

            # With Yan's trick bu updating dvp
            # self.update_oueis_retularized_vp_nad(dvp=dvp)

            # f, ax = plt.subplots(1, 1, dpi=210)
            # ax.set_ylim(-2,0.2)
            # plot1d_x(self.vp_grid, self.molecule.Vpot, ax=ax, label="vp", color="black", title=str(int(scf_step))+ '-')
            # plot1d_x(self.vp_Hext_nad, self.molecule.Vpot, dimmer_length=4.522, ax=ax, label="Hext", ls='--')
            # plot1d_x(self.vp_xc_nad, self.molecule.Vpot, ax=ax, label="xc", ls='--')
            # plot1d_x(self.vp_kin_nad, self.molecule.Vpot, ax=ax, label="kin", ls='--')
            # ax.legend()
            # f.savefig("H2p svd_step vs[" + vstype + "]Q["+Qtype+"]""-" + str(int(scf_step)))
            # plt.close(f)

            self.fragments_scf_1basis(1000, vp_fock=True)

            if beta < 1e-7:
                print("Break because even small step length can not improve.")
                break
            elif len(self.drho_conv) >= 5:
                if np.std(self.drho_conv[-4:]) < a_rho_var and vp_change < vp_norm_conv:
                    print("Break because rho and vp do not update for 5 iterations.")
                    break
            elif old_rho_conv < 1e-4:
                print("Break because rho difference (cost) is small.")
                break
            # elif scf_step == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded for vp.")
                # print("Maximum number of SCF cycles exceeded for vp.")
        # update current vp.
        self.update_oueis_retularized_vp_nad(Qtype=Qtype, vstype=vstype,
                                             vp_Hext_decomposition=False if (vp_nad_iter is None) else True)
        return

    def find_vp_response_crude(self, maxiter=21, beta=None, atol=1e-7, guess=None):
        """
        This is a very initial and naive idea. It works to some extent but very not stable.
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

    def find_vp_all96(self, vp_maxiter, scf_maxiter, guess=None, rtol=1e-3, separation_cutoff=None, hard_cutoff=None):
        """
        vp = vp_non-local = vp_all96. Total scf iteration max = vp_maxiter*scf_maxiter*num_fragments + entire system scf
        :param vp_maxiter: maximum num of vp update iteration needed.
        :param vp_maxiter: maximum num of scf update iteration needed.
        :param guess: Initial guess of vp.
        :param rtol: Relative ALL96 energy difference as the convergence criteria.
        :para hard_cutoff: the main idea of this cutoff is that: the density and grad of density is not accurate on a basis
              set around the nucleus. e.g. for two fragments systems, the usual cutoff given by ALL96 will be True for
              fragmentB around the nucleus of fragmentA, which is of course not true. This will introduce singularities.
              Right now if it is True, then there is a hard cutoff at the yz plane.
              seperation_cutoff (removed): a very crude cutoff to avoid singularity: if a piece |r1-r2| is smaller than this value,
              it will be neglected in the integral. The reason is that Gaussian basis sets are bad around the nucleus.
              Thus the cutoff of one fragment will not kill the density around the other fragments' nucleus.
              I designed this hard cutoff to overcome this. A loose upper-bound for seperation_cutoff is the seperation between
              the two nucleus.
        :return:
        """
        # Find some a vp with density difference method.
        self.find_vp_densitydifference(49, 2)
        # self.fragments_scf(100)

        all96_e_old = np.inf
        vp_fock_all96_old = 0.0
        for vp_step in range(1, vp_maxiter+1):
            self.get_density_sum()
            # Initial vp_all96
            all96_e, vp_all96, vp_fock_all96 = self.vp_all96(separation_cutoff=separation_cutoff)

            # Check if vp_all96 consists with vp_fock_all96
            vp_fock_temp = self.molecule.grid_to_fock(vp_all96)

            assert np.allclose(vp_fock_temp, vp_fock_all96), \
                'vp_all96 does not consists with vp_fock_all96.'

            vp_fock_total = self.vp_fock[0]
            vp_fock_total.np[:] += vp_fock_all96
            self.vp_fock = [vp_fock_total,vp_fock_total]

            self.fragments_scf(scf_maxiter, vp_fock=self.vp_fock)

            f, ax = plt.subplots(1, 1)
            plot1d_x(vp_all96, self.molecule.Vpot, dimmer_length=4, ax=ax, title="He2 svwn sp2")
            f.savefig("He2 svwn sp2")
            plt.close(f)

            if abs((all96_e_old - all96_e) / all96_e) < rtol \
                    and \
                    np.linalg.norm(vp_fock_all96_old - vp_fock_all96) < rtol:
                print("ALL96 Energy Converged:", all96_e)
                print("Iteration % i, ALL96 E %.14f, ALL96 E difference %.14f" % (
                vp_step, all96_e, abs((all96_e_old - all96_e) / all96_e)))
                break
            print("Iteration % i, ALL96 E %.14f, ALL96 E difference %.14f" % (vp_step, all96_e, abs((all96_e_old - all96_e) / all96_e)))
            all96_e_old = all96_e
            vp_fock_all96_old = vp_fock_all96
        f, ax = plt.subplots(1, 1)
        plot1d_x(vp_all96, self.molecule.Vpot, dimmer_length=4, ax=ax, title="He2 svwn sp2 ")
        # f.savefig("He2 svwn sp2 " + str(int(seperation_cutoff*2*100)))
        plt.close(f)
        return all96_e, vp_all96, vp_fock_all96

    def vp_all96(self, beta=6, separation_cutoff=None, hard_cutoff=None):
        """
        Return vp on grid and vp_fock on the basis for a specific density.
        :para seperation_cutoff: a very crude cutoff to avoid singularity: if a piece |r1-r2| is smaller than this value,
        it will be neglected in the integral. The reason is that Gaussian basis sets are bad around the nucleus.
        Thus the cutoff of one fragment will not kill the density around the other fragments' nucleus.
        I designed this hard cutoff to overcome this. A loose upper-bound for seperation_cutoff is the seperation between
        the two nucleus.
        """

        C = -6.0 / 4.0 / (4 * np.pi) ** 1.5

        vp = np.zeros_like(self.molecule.Vpot.get_np_xyzw()[-1])
        vp_fock = np.zeros_like(self.fragments[0].Da.np)

        points_func = self.molecule.Vpot.properties()[0]

        points_func.set_deriv(2)

        w1_old = 0

        all96_e = 0.0

        # First loop over the outer set of blocks
        for l_block in range(self.molecule.Vpot.nblocks()):
            #     for l_block in range(70, Vpot.nblocks()):
            # Obtain general grid information
            l_grid = self.molecule.Vpot.get_block(l_block)
            l_w = np.array(l_grid.w())
            l_x = np.array(l_grid.x())
            l_y = np.array(l_grid.y())
            l_z = np.array(l_grid.z())
            l_npoints = l_w.shape[0]

            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())

            # Compute phi!
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]
            # Build a local slice of D
            lD1 = self.fragments[0].Da.np[(l_lpos[:, None], l_lpos)]

            # Copmute block-rho and block-gamma
            rho1 = 2.0 * np.einsum('pm,mn,pn->p', l_phi, lD1, l_phi, optimize=True)

            total_rho1 = 2.0 * np.einsum('pm,mn,pn->p', l_phi, lD1 + self.fragments[1].Da.np[(l_lpos[:, None], l_lpos)], l_phi,
                                         optimize=True)

            # 2.0 for Px D P + P D Px, 2.0 for non-spin Density
            rho_x1 = 4.0 * np.einsum('pm,mn,pn->p', l_phi, lD1, l_phi_x, optimize=True)
            rho_y1 = 4.0 * np.einsum('pm,mn,pn->p', l_phi, lD1, l_phi_y, optimize=True)
            rho_z1 = 4.0 * np.einsum('pm,mn,pn->p', l_phi, lD1, l_phi_z, optimize=True)
            gamma1 = rho_x1 ** 2 + rho_y1 ** 2 + rho_z1 ** 2

            # The integral cutoff.
            l_local_w_homo = gamma1 ** 0.5 <= 2 * beta * ((9 * np.pi) ** (-1.0 / 6.0)) * (rho1 ** (7.0 / 6.0))
            l_local_w_rho = rho1 > 1e-17
            l_local_w = l_local_w_homo * l_local_w_rho

            if not np.any(l_local_w):
                w1_old += l_npoints
                continue

            w2_old = 0
            l_integrant = np.zeros_like(rho1)
            dvp_l = np.zeros(l_npoints)
            # Loop over the inner set of blocks
            for r_block in range(self.molecule.Vpot.nblocks()):
                r_grid = self.molecule.Vpot.get_block(r_block)
                r_w = np.array(r_grid.w())
                r_x = np.array(r_grid.x())
                r_y = np.array(r_grid.y())
                r_z = np.array(r_grid.z())
                r_npoints = r_w.shape[0]

                if hard_cutoff is not None and np.all((l_x[:, None] * r_x) >= 0):
                    w2_old += r_npoints
                    continue

                points_func.compute_points(r_grid)
                r_lpos = np.array(r_grid.functions_local_to_global())

                # Compute phi!
                r_phi = np.array(points_func.basis_values()["PHI"])[:r_npoints, :r_lpos.shape[0]]
                r_phi_x = np.array(points_func.basis_values()["PHI_X"])[:r_npoints, :r_lpos.shape[0]]
                r_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:r_npoints, :r_lpos.shape[0]]
                r_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:r_npoints, :r_lpos.shape[0]]

                # Build a local slice of D
                lD2 = self.fragments[1].Da.np[(r_lpos[:, None], r_lpos)]

                total_rho2 = 2.0 * np.einsum('pm,mn,pn->p', r_phi, self.fragments[0].Da.np[(r_lpos[:, None], r_lpos)] + lD2, r_phi,
                                             optimize=True)

                # Copmute block-rho and block-gamma
                rho2 = 2.0 * np.einsum('pm,mn,pn->p', r_phi, lD2, r_phi, optimize=True)
                # 2.0 for Px D P + P D Px, 2.0 for non-spin Density
                rho_x2 = 4.0 * np.einsum('pm,mn,pn->p', r_phi, lD2, r_phi_x, optimize=True)
                rho_y2 = 4.0 * np.einsum('pm,mn,pn->p', r_phi, lD2, r_phi_y, optimize=True)
                rho_z2 = 4.0 * np.einsum('pm,mn,pn->p', r_phi, lD2, r_phi_z, optimize=True)
                gamma2 = rho_x2 ** 2 + rho_y2 ** 2 + rho_z2 ** 2

                # The integrate cutoff.
                r_local_w_homo = gamma2 ** 0.5 <= 2 * beta * ((9 * np.pi) ** (-1.0 / 6.0)) * (rho2 ** (7.0 / 6.0))
                r_local_w_rho = rho2 > 1e-17
                r_local_w = r_local_w_homo * r_local_w_rho
                #           r_local_w = r_local_w_homo

                if not np.any(r_local_w):
                    w2_old += r_npoints
                    continue

                # Build the distnace matrix
                R2 = (l_x[:, None] - r_x) ** 2
                R2 += (l_y[:, None] - r_y) ** 2
                R2 += (l_z[:, None] - r_z) ** 2
                R2 += 1e-34
                if hard_cutoff is not None:
                    R_hard_cut = (l_x[:, None] * r_x) < 0
                    # R6inv = R2 ** -3 * R_hard_cut
                    R6inv = R2 ** -3 * R_hard_cut
                elif separation_cutoff is not None:
                    R6inv = R2 ** -3 * (R2 > separation_cutoff**2)
                else:
                    R6inv = R2 ** -3
                
                # vp calculation.
                # Add vp for fragment 1
                dvp_l += np.sum(rho2
                                / (np.sqrt(rho1[:, None]) + np.sqrt(rho2) + 1e-34) ** 2
                                * R6inv * r_local_w * r_w, axis=1
                                ) * np.sqrt(rho1) / (total_rho1 + 1e-34) * 0.5 * l_local_w

                # Add vp for fragment 2
                dvp_r = np.sum(rho1[:, None]
                               / (np.sqrt(rho1[:, None]) + np.sqrt(rho2) + 1e-34) ** 2
                               * R6inv * l_local_w[:, None] * l_w[:, None], axis=0
                               ) * np.sqrt(rho2) / (total_rho2 + 1e-34) * 0.5 * r_local_w
                vp[w2_old:w2_old + r_npoints] += dvp_r

                # E calculation
                r_integrant = np.sqrt(rho1[:, None] * rho2) / (np.sqrt(rho1[:, None]) + np.sqrt(rho2) + 1e-34) * R6inv
                l_integrant += np.sum(r_integrant * r_local_w * r_w, axis=1)

                # Add vp_fock for fragment 2
                vp_fock[(r_lpos[:, None], r_lpos)] += np.einsum("p,p,pa,pb->ab", r_w, dvp_r,
                                                                r_phi, r_phi, optimize=True)
                w2_old += r_npoints

            vp[w1_old:w1_old + l_npoints] += dvp_l
            # Add vp_fock for fragment 1
            vp_fock[(l_lpos[:, None], l_lpos)] += np.einsum("p,p,pa,pb->ab", l_w, dvp_l, l_phi,
                                                            l_phi, optimize=True)
            w1_old += l_npoints
            # E calculation
            all96_e += C * np.sum(l_integrant * l_local_w * l_w)

        vp_fock = 0.5 * (vp_fock + vp_fock.T)
        vp *= C
        vp_fock *= C
        if np.any(np.abs(vp) > 1e3):
            print("Singulartiy vp %f" % np.linalg.norm(vp))
        return all96_e, vp, vp_fock

def plot1d_x(data, Vpot, dimmer_length=None, title=None,
             ax=None, label=None, color=None, ls=None, lw=None):
    """
    Plot on x direction
    :param data: Any f(r) on grid
    """
    x, y, z, w = Vpot.get_np_xyzw()
    # filter to get points on z axis
    mask = np.isclose(abs(y), 0, atol=1E-11)
    mask2 = np.isclose(abs(z), 0, atol=1E-11)
    order = np.argsort(x[mask & mask2])
    if ax is None:
        f1 = plt.figure(figsize=(16, 12), dpi=160)
        # f1 = plt.figure()
        plt.plot(x[mask & mask2][order], data[mask & mask2][order],
                 label=label, color=color, ls=ls, lw=lw)
    else:
        ax.plot(x[mask & mask2][order], data[mask & mask2][order],
                label=label, color=color, ls=ls, lw=lw)
    if dimmer_length is not None:
        plt.axvline(x=dimmer_length/2.0, ls="--", lw=0.7, color='r')
        plt.axvline(x=-dimmer_length/2.0, ls="--", lw=0.7, color='r')
    if title is not None:
        if ax is None:
            plt.title(title)
        else:
            # f1 = plt.figure(num=fignum, figsize=(16, 12), dpi=160)
            ax.set_title(title)
    if ax is None:
        plt.show()
