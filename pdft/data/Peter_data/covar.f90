!This program originally did an AR1 and AR2 analysis also, but I removed that once I realized that AR2
!does not have 3 parameters, r1, r2 and a1, needed to reproduce in the correlation function which is
!a sum of 2 exponentials, rho(i) = a1*r1^i + a2*r2^i,  (a1+a2=1).
!It has only 2 parameters, and a2 is negative, whereas in actual QMC energies a1 and a2 are positive.
!So, when applied to actual series, AR2 gives almost the same esimate as AR1, and both are underestimates,
!whereas my sum of 2 exponentials gives a larger estimate than 1 exponential and both are larger than AR1, AR2 estimates.
!So, now the program just does the following:
!1) It calculates the usual errors from blocking, which of course are gross underestimates for low blocking levels.
!2) It fits these errors to Eq. 15 of my notes, to evaluate r1, r2, and a1.
!3) It evaluates Eq. 11 to find an unbiased estimate of the error.
!It can get statistics on how well it does by breaking up a long series into a bunch of smaller ones.

! To be done:
! 1) put in DMC wts.  (The wts for different blocking levels are in func.)
! 2) figure out why for small data sizes, the rho from blocking is a bit smaller than that from AR1 when the data is AR1

module data_quench
  implicit none
  integer, parameter :: r8b =  SELECTED_REAL_KIND(15) ! 8 bytes = 64 bits (double precision what we usually use)
  integer icalls, weighting
  real(r8b) r1, r2, a1
  character*8 fit_what
end module data_quench
!-------------------------------------------------------------------------

module errors
! To be done:
! 1) put in DMC wts.  (The wts for different blocking levels are in func.)
! 2) figure out why for small data sizes, the rho from blocking is a bit smaller than that from AR1 when the data is AR1

implicit none
integer, parameter :: r8b =  SELECTED_REAL_KIND(15) ! 8 bytes = 64 bits (double precision what we usually use)
integer N_MC, N_s, N_b, nparm
integer, parameter :: M_BLOCKING_LEVEL=50
real(r8b) :: ave(0:M_BLOCKING_LEVEL), err(0:M_BLOCKING_LEVEL)

public :: read_input, func, idum_jacobian, err

! If the autocorrelation function (ACF) is a single exponential, then r1 is the ratio of the ACF at subsequent lags.
! If the autocorrelation function (ACF) is a a sum of 2 exponentials with linear coefficients a1 and (1-a1),
! r1 is no longer the lag 1 ACF unless a1=1.

contains

function TT_c_fn(N,r) ! Eq. 14
integer N
real(r8b) TT_c_fn, r
TT_c_fn = 1+2*(r/(1-r))*(1-(1-r**N)/(N*(1-r)))
end function TT_c_fn

function TT_corr_fn(N,r1,r2,a) ! Eq. 13
integer N
real(r8b) TT_corr_fn, r1,r2,a
TT_corr_fn = a*TT_c_fn(N,r1) + (1-a)*TT_c_fn(N,r2)
end function TT_corr_fn

function N_eff(N,Tcorr)
integer N
real(r8b) N_eff, Tcorr
N_eff = N/Tcorr
end function N_eff
!-------------------------------------------------------------------------

subroutine read_input
use data_quench, only: r1, r2, a1, weighting, fit_what
integer, parameter :: MDATA=12000000, MLAG=10 ! 10000
integer i_MC,lag,nskip,N_MC_r,iset,nset,i_blocking_level,iskip
real(r8b) wt(MDATA),energies(MDATA),gamma(0:MLAG),rho(1:MLAG),aver(0:MLAG,2),del(2),vn1(0:MLAG,2), &
variance, variance_del, variance_ave, variance_vn1, variance_sigma, &
err_blocking, err_blocking_del, err_blocking_ave, err_blocking_vn1, err_blocking_sigma, &
err_formula, err_formula_del, err_formula_ave, err_formula_vn1, err_formula_sigma, &
r1_del, r1_ave, r1_vn1, r1_sigma, &
skip
character*80 energy_filename, covariance_filename

write(6,'(''Input energy_filename, covariance_filename'')')
read(5,*) energy_filename, covariance_filename
write(6,'(''Input nskip, N_MC, nparm'')')
read(5,*) nskip, N_MC, nparm
write(6,'(''nskip, N_MC, nparm ='',2i9,i4)') nskip, N_MC, nparm
if(N_MC.gt.MDATA) stop 'N_MC>MDATA'
read(5,*) weighting, fit_what
write(6,'(''weighting, fit_what ='',i3,x,a)') weighting, fit_what
write(6,'(''weighting = 0 => fit upto some blocking level without wts,'', &
     &   /,''          = 1 => fit all blocking levels with weight sqrt(nblk-1)'', &
     &   /,''          = 2 => fit all blocking levels with weight (nblk-1)'', &
     &   /,''fit_what  = "variance" => fit RHS of Eq. 15 in beyond_FN.tex'', &
     &   /,''          = "error" => fit sqrt of RHS of Eq. 15 in beyond_FN.tex'')')

open(1,file=trim(energy_filename))
open(2,file=trim(covariance_filename))

do i_MC=1,nskip
  read(1,*)
enddo

!write(6,'(''# aver variance T_corr_AR1 T_corr_AR2  err_AR1  err_AR2  err_blocking, err_formula'')')

err_blocking_ave=0; err_blocking_vn1=0
variance_ave=0; variance_vn1=0

! Calculate the covariance, gamma, and the correlation function rho
! vn1 is the variance.  Its expectation value does not depend on its indices.  They are there just to have error cancelation.
nset=0
do iset=1,100000
!do iset=1,2075
 
  aver=0; gamma=0; vn1=0
  do i_MC=1,N_MC
    !read(1,*,end=99) energies(i_MC)
    read(1,*,end=99) iskip,skip,skip,energies(i_MC)
    do lag=0,MLAG
      if(i_MC.ge.lag+1) then
        del(1)=energies(i_MC-lag)-aver(lag,1)
        aver(lag,1)=aver(lag,1)+del(1)/(i_MC-lag)                     ! average using first N_MC-lag pts
        del(2)=energies(i_MC)-aver(lag,2)
        aver(lag,2)=aver(lag,2)+del(2)/(i_MC-lag)                     ! average using last N_MC-lag pts
        gamma(lag)=gamma(lag)+(energies(i_MC-lag)-aver(lag,1))*del(2) ! covariance at lag
        vn1(lag,1)=vn1(lag,1)+(energies(i_MC-lag)-aver(lag,1))*del(1) ! variance using first N_MC-lag pts
        vn1(lag,2)=vn1(lag,2)+(energies(i_MC)-aver(lag,2))*del(2)     ! variance using last N_MC-lag pts
      endif
    enddo
  enddo
  nset=nset+1
  write(6,'(/,''iset='',i8)') iset
 
  do lag=0,MLAG
    gamma(lag)=gamma(lag)/(N_MC-lag)
    vn1(lag,1)=vn1(lag,1)/(N_MC-lag)
    vn1(lag,2)=vn1(lag,2)/(N_MC-lag)
  enddo
  variance=vn1(0,1) ! variance using all pts
 
  do lag=1,MLAG
    rho(lag)=gamma(lag)/sqrt(vn1(lag,1)*vn1(lag,2)) ! correlation function at lag
  enddo

! write(6,'(''gamma(0)/variance,gamma(0),variance'',f11.8,2es16.8)') gamma(0)/variance,gamma(0),variance

! write(6,'(''aver1='',11es14.6)') aver(0:min(MLAG,10),1)
! write(6,'(''aver2='',11es14.6)') aver(0:min(MLAG,10),2)
! write(6,'(''autocovar='',11es14.6)') gamma(0:min(MLAG,10))
! write(6,'(''autocorrel='',11f10.6)') rho(1:min(MLAG,10))
 
  write(2,'(''! Using energies from '',a)') trim(energy_filename)
  write(2,'(''! nskip, N_MC ='',2i9)') nskip, N_MC
  write(2,'(''! aver1='',11es14.6)') aver(0:min(MLAG,10),1)
  write(2,'(''! aver2='',11es14.6)') aver(0:min(MLAG,10),2)
  write(2,'(''! autocovar='',11es14.6)') gamma(0:min(MLAG,10))
  write(2,'(''! lag  autocorrel'')')
  write(2,'(''0 1.'',/,(i5,f13.9))') (lag,rho(lag),lag=1,min(MLAG,N_MC))
 
  if(rho(1).lt.0._r8b .or. rho(1).gt.1._r8b) then
    write(6,'(''rho(1).gt.1'',es16.8)') rho(1)
    stop 'Lag 1 correlation function rho(1) < 0 or >1'
  endif
 

! Call recursive function to compute errors for various blocking levels, using formula that would be correct if Tcorr=1.
! So, the estimated errors go up with blocking level, and eventually become noisy because uncertainty of error is large.
! Warning: For the moment we are just all QMC wts=1.
  wt=1; i_blocking_level=0
  N_MC_r=N_MC
  call blocking(wt, energies, N_MC_r, i_blocking_level)

! Find the true error by fitting the above to the theoretical formula
  call true_error(i_blocking_level, err_blocking, err_formula, r1, r2, a1)

  err_blocking_del=err_blocking-err_blocking_ave
  err_blocking_ave=err_blocking_ave+err_blocking_del/iset
  err_blocking_vn1=err_blocking_vn1+(err_blocking-err_blocking_ave)*err_blocking_del

  err_formula_del=err_formula-err_formula_ave
  err_formula_ave=err_formula_ave+err_formula_del/iset
  err_formula_vn1=err_formula_vn1+(err_formula-err_formula_ave)*err_formula_del

  variance_del=variance-variance_ave
  variance_ave=variance_ave+variance_del/iset
  variance_vn1=variance_vn1+(variance-variance_ave)*variance_del

  r1_del=r1-r1_ave
  r1_ave=r1_ave+r1_del/iset
  r1_vn1=r1_vn1+(r1-r1_ave)*r1_del

! write(6,'(''    alpha1      alpha2      omega1      omega2      lambda1   Re(lambda2)     a1         a2          sum1      sum2   sum(AR2)  sum(AR1)'')')
! write(6,'(''aver, variance, TT_corr_AR1, TT_corr_AR2, TT_corr_blocking, err_AR1, err_AR2, err_blocking, err_formula'',es14.6,es13.6,3f7.1,9es9.2)') aver(0,1), variance, TT_corr_AR1, TT_corr_AR2, TT_corr(r1), err_AR1, err_AR2, err_blocking, err_formula
  write(6,'(''aver, variance, TT_corr, err_blocking, err_formula'',es14.6,es13.6,f7.1,9es9.2)') aver(0,1), variance, TT_corr_fn(N_MC,r1,r2,a1), err_blocking, err_formula

enddo

99 continue

!When there are several data sets (nset>1) one can calculate the uncertainty of the estimated errors.
r1_sigma=sqrt(r1_vn1/(nset-1))
err_blocking_sigma=sqrt(err_blocking_vn1/(nset-1))
err_formula_sigma=sqrt(err_formula_vn1/(nset-1))
write(6,'(''r1_ave, r1_sigma='',9f9.6)') r1_ave, r1_sigma
write(6,'(''variance_ave, variance_sigma='',9es14.6)') variance_ave, variance_sigma
write(6, '(''#N_MC, nset='', 2i8, '' err_blocking_ave, err_blocking_sigma, err_formula_ave, err_formula_sigma='',9f10.6)') &
N_MC, nset, err_blocking_ave, err_blocking_sigma, err_formula_ave, err_formula_sigma

return
end subroutine read_input
!-------------------------------------------------------------------------
recursive subroutine blocking(wt, energies, N_MC_r, i_blocking_level)
! Note, energies and N_MC_r get changed, the latter because of recursion, but N_MC does not.

real(r8b), intent(inout) :: wt(N_MC), energies(N_MC)
integer, intent(inout) :: N_MC_r, i_blocking_level
integer :: i_MC
real(r8b) :: wt_sum, del

! This is the end of the recursion.
if(N_MC_r.le.1) then
  return
endif

! Use Welford to calculate weighted average and error.
! Warning, although the following code has wts, we are presently setting all wts to 1.  And the Welford for AR does not have wts yet.  Not to be confused with the wts. in func for different blocking levels.
wt_sum=0; ave(i_blocking_level)=0; err(i_blocking_level)=0
do i_MC=1,N_MC_r
  wt_sum=wt_sum+wt(i_MC)
  del=energies(i_MC)-ave(i_blocking_level)
  ave(i_blocking_level)=ave(i_blocking_level)+wt(i_MC)*del/wt_sum
  err(i_blocking_level)=err(i_blocking_level)+wt(i_MC)*(energies(i_MC)-ave(i_blocking_level))*del
!write(6,*) wt_sum, del, ave(i_blocking_level), err(i_blocking_level)
!If N_MC is not a power of 2, we may get an odd number of pts, i_MC, at some blocking levels.
!The elseif below takes care of this.
  if(mod(i_MC,2).eq.0) then
    energies(i_MC/2)=(wt(i_MC-1)*energies(i_MC-1)+wt(i_MC)*energies(i_MC))/(wt(i_MC-1)+wt(i_MC))
    wt(i_MC/2)=wt(i_MC-1)+wt(i_MC)
  elseif(i_MC.eq.N_MC_r) then ! last point if total number is odd
    energies((i_MC+1)/2)=energies(i_MC)
    wt((i_MC+1)/2)=wt(i_MC)
  endif
enddo
err(i_blocking_level)=sqrt(err(i_blocking_level)/(wt_sum*(N_MC_r-1)))

!write(6,'(''i_blocking_level, N_MC_r, wt_sum, ave, err, T_corr_est/N_s='',i3,i8,3es16.8,f7.3)') i_blocking_level, N_MC_r, wt_sum, ave(i_blocking_level), err(i_blocking_level), (err(i_blocking_level)/err(0))**2/2**i_blocking_level

i_blocking_level=i_blocking_level+1; N_MC_r=(N_MC_r+1)/2
call blocking(wt, energies, N_MC_r, i_blocking_level)

end subroutine blocking
!-------------------------------------------------------------------------
subroutine true_error(i_blocking_level, err_blocking, err_formula, r1, r2, a1)
! If the autocorrelation function is a single decaying exponential then
! r1 is the ratio of the autocorrelation function at one time step to the one at the previous time step.

use data_quench, only: weighting

real(r8b), intent(out) :: err_blocking, err_formula, r1, r2, a1
integer, intent(inout) :: i_blocking_level
integer :: i_blocking_level_best, j_blocking_level
real(r8b) :: TT_corr_blocking_biased

  i_blocking_level_best=i_blocking_level
  do j_blocking_level=0,i_blocking_level-1
    if(err(j_blocking_level+1).lt.err(j_blocking_level) .or. &
      err(j_blocking_level+1)-err(j_blocking_level).lt.2*(err(j_blocking_level+1)/sqrt(2*(real(N_MC,r8b)/2**(j_blocking_level+1))-1)-err(j_blocking_level)/sqrt(2*(real(N_MC,r8b)/2**(j_blocking_level))-1))) goto 5
  enddo

5 i_blocking_level_best=j_blocking_level
! TT_corr_biased cannot be greater than nstep (N_s), so if that is violated go to an earlier blocking level for which it holds.
! The problem is that when we have 2^11 pts or less, there are a few data sets where this is not satisfied for any blocking level.
  do j_blocking_level=i_blocking_level_best,1,-1
    if((err(j_blocking_level)/err(0))**2 .lt. 2**j_blocking_level) then
      goto 6
    else
      write(6,'(/,''Warning: TT_corr_biased > N_step'',/)')
    endif
  enddo

! Now we have found the optimal blocking level according to Peter's criterion, we can either simply take the
! corresponding error as a reasonable estimate.
! Instead we do better by doing either an unweighted fit upto this point and extrapolating to N_s = N_MC,
! or else we do a weighted fit with all the data, which is preferable.
6 i_blocking_level_best=j_blocking_level
  err_blocking=err(i_blocking_level_best)
  write(6,'(''i_blocking_level_best, err_blocking='',i3,es16.8)') i_blocking_level_best, err_blocking
  TT_corr_blocking_biased=(err_blocking/err(0))**2
  N_s=2**i_blocking_level_best
  N_b=(N_MC-1)/N_s+1 ! Note when N_MC is not a power of 2 we are presently choosing to round up rather than round down (though this may not be the better choice)
! Initial values of 2 fit parameters
  r1=(TT_corr_blocking_biased-1)/(TT_corr_blocking_biased+1)
  write(6,'(''TT_corr_blocking_biased, r1_initial'',f10.2,f8.5)') TT_corr_blocking_biased, r1
  if(nparm.eq.1) then
    a1=1
  elseif(nparm.eq.3) then
    r2=r1**.8
    r1=r1**2
    a1=.5
  endif
  if(r1.le.0.1) write(6,'(''Warning: i_blocking_level_best, err_blocking, err_blocking, err(0)='',i5,9es12.4)') i_blocking_level_best, err_blocking, err_blocking, err(0)

! Fit to get r1, r2, a1
  if(weighting==0) then
    call do_fit(i_blocking_level_best+1,nparm) ! arguments are ndata, nparm
  else
    call do_fit(i_blocking_level,nparm) ! arguments are ndata, nparm
  endif
  do j_blocking_level=0,i_blocking_level
    N_s=2**j_blocking_level
    N_b=(N_MC-1)/N_s+1
    err_formula=err_biased(N_MC,N_s,N_b,err(0),r1,r2,a1)
    if(j_blocking_level.eq.i_blocking_level) then
      write(6,'(''j_blocking_level, err, err_formula='',i3,11x,es11.3,'' <-- This is the final result'')') j_blocking_level, err_formula
    elseif(j_blocking_level.eq.0 .or. j_blocking_level.eq.i_blocking_level_best) then
      write(6,'(''j_blocking_level, err, err_formula='',i3,2es11.3,es10.2,f7.3'' <--'')') j_blocking_level, err(j_blocking_level), err_formula, err_formula-err(j_blocking_level), (err(j_blocking_level)/err(0))**2/2**j_blocking_level
    else
      write(6,'(''j_blocking_level, err, err_formula='',i3,2es11.3,es10.2,f7.3)') j_blocking_level, err(j_blocking_level), err_formula, err_formula-err(j_blocking_level), (err(j_blocking_level)/err(0))**2/2**j_blocking_level
    endif
  enddo

  return

end subroutine true_error
!-------------------------------------------------------------------------
function err_biased(N_MC,N_s,N_b,err,r1,r2,a1)
integer, intent(in) :: N_MC,N_s,N_b
real(r8b), intent(in) :: err, r1, r2, a1
real(r8b) err_biased, tcorr_Ns, tcorr_MC

  tcorr_Ns=TT_corr_fn(N_s,r1,r2,a1)
  tcorr_MC=TT_corr_fn(N_MC,r1,r2,a1)

  if(N_s.eq.N_MC) then
    err_biased=err*sqrt(real(N_MC-1,r8b)*tcorr_MC/(N_MC-tcorr_MC)) ! Eq. 11 (the unbiased error, despite the name)
  else
    err_biased=err*sqrt(((real(N_MC,r8b)*(N_MC-1))/(real(N_b-1,r8b)*(N_MC-tcorr_MC))) * (tcorr_Ns/N_s-tcorr_MC/N_MC)) ! Eq. 15
  endif

end function err_biased
!-------------------------------------------------------------------------
subroutine do_fit(ndata,nparm)

use data_quench, only: icalls, r1, r2, a1
integer, intent(in) :: ndata, nparm
integer :: noutput=1, nstep=1000, ipr=-2, nanalytic=0, ibold=4
real(r8b) :: pmarquardt=1.d-6, tau=2, epsp=1.d-9, epsg, epsch2, rot_wt=1., eps_diff=1.d-15
real(r8b) :: err2, parm(nparm), diff(ndata)
logical :: cholesky=.true., converg
character*10 mesg

write(6,'(''ndata='',i9)') ndata

epsg=0.001_r8b*epsp; epsch2=epsp
icalls=0

parm(1)=acos(sqrt(r1))
if(nparm.eq.3) then
  parm(2)=acos(sqrt(r2))
  parm(3)=acos(sqrt(a1))
endif
call quench(func,idum_jacobian,nanalytic,parm,pmarquardt,tau,noutput,nstep,ndata,nparm, &
  ipr,diff,err2,epsg,epsp,epsch2,converg,mesg,ibold,cholesky,rot_wt,eps_diff)
r1=(cos(parm(1)))**2
if(nparm.eq.3) then
  r2=(cos(parm(2)))**2
  a1=(cos(parm(3)))**2
endif
if(nparm.eq.1) then
  write(6,'(''r1='', 9f10.6)') r1
  write(6,'(''correlation length='',f12.3)') -1/log(r1)
else
  write(6,'(''r1, r2, a1='',9f10.6)') r1, r2, a1
  write(6,'(''correlation lengths, a1='',2f12.3,f8.3)') -1/log(r1), -1/log(r2), a1
endif

if(converg) then
  write(6,'(''chisq='',d12.6,i6,'' func evals, convergence: '',a10)') err2,icalls,mesg
 else
  write(6,'(''chisq='',d12.6,i6,'' func evals, no convergence'')') err2,icalls
endif

return
end subroutine do_fit
!-------------------------------------------------------------------------
function func(ndata,nparm,parm,diff,iflag)

use data_quench, only: icalls, r1, r2, a1, weighting
integer, intent(in) :: ndata, nparm, iflag
real(r8b), intent(in) :: parm
real(r8b), intent(out) :: diff
dimension parm(nparm), diff(ndata)
integer i
!real(r8b) func, err_formula, Tcorr_Ns, Tcorr_MC, sigmasq_x
real(r8b) func, err_formula
common /quenchsim_pr/ ipr_com,called_by_qa
integer ipr_com
logical called_by_qa

if(iflag.eq.0) then

  icalls=icalls+1
  r1=(cos(parm(1)))**2
  if(nparm.eq.3) then
    r2=(cos(parm(2)))**2
    a1=(cos(parm(3)))**2
  endif
  if(called_by_qa .and. nparm.eq.1) write(6,'(''r1='', 9f20.16)') r1
  if(called_by_qa .and. nparm.eq.3) write(6,'(''r1, r2, a1'',9f10.6)') r1, r2, a1

  do i=1,ndata
    N_s=2**(i-1)
    N_b=(N_MC-1)/N_s+1
!   diff(i)=0
!   if(nparm.le.2 .and. r1.ge.0.9999_r8b .or. (nparm.eq.3 .and.  (r1.ge.0.9999_r8b .or. r2.ge.0.9999_r8b .or. a1.lt.0._r8b .or. a1.gt.1._r8b))) then
!     if(r1.ge.0.9999_r8b) diff(i)=9.d99*r1
!     if(nparm.eq.3 .and. r2.ge.0.9999_r8b) diff(i)=diff(i)+9.d99*r2
!     if(nparm.eq.3 .and. a1.lt.0._r8b) diff(i)=diff(i)-9.d99*a1
!     if(nparm.eq.3 .and. a1.gt.1._r8b) diff(i)=diff(i)+9.d99*a1
!   else
      err_formula=err_biased(N_MC,N_s,N_b,err(0),r1,r2,a1)
! Either use 1st several blocking levels and no weight, or
! use all the blocking levels and weight with either sqrt(nblk-1) or (nblk-1)
      if(weighting==0) then
        diff(i)=err_formula-err(i-1)
      elseif(weighting==1) then
        diff(i)=(err_formula-err(i-1))*sqrt(real(2**(ndata-i+1)-1))
      elseif(weighting==2) then
        diff(i)=(err_formula-err(i-1))*(real(2**(ndata-i+1)-1))
      endif
      if(i.le.2) write(6,'(''i, diff(i), err_formula-err(i-1)'',i5,9es12.4)') i, diff(i), err_formula, err(i-1)
!   endif
  enddo

else

  func=0
  do i=1,ndata
    func=func+diff(i)**2
    !write(6,'(9es15.6)') err(i),err(i)+diff(i)
  enddo
! if(nparm.eq.1) write(6,'(''r1, func='',9es24.16)') r1, func
! if(nparm.eq.3) write(6,'(''r1, r2, a1, func='',9es24.16)') r1, r2, a1, func

endif

return
end function func
!-----------------------------------------------------------------------
subroutine idum_jacobian
stop 'dum_jac entered'
return
end subroutine idum_jacobian
!-----------------------------------------------------------------------

end module errors
!-----------------------------------------------------------------------
program covar

use errors, only: read_input
integer, parameter :: r8b =  SELECTED_REAL_KIND(15) ! 8 bytes = 64 bits (double precision what we usually use)

call read_input

end program covar
