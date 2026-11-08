!> Équation intégro-différentielle de Prandtl (ligne portante), résolue en parallèle.
!>
!>   alpha(y) = 2 Gamma(y) / (a0 V c(y))  +  (1/(4 pi V)) PV int_{-b/2}^{b/2} Gamma'(eta)/(y-eta) d eta
!>
!> Discrétisation : N panneaux uniformes de largeur h, Gamma constant par panneau (g = Gamma/V),
!> noyau de Cauchy évalué aux centres des panneaux. Gamma' est alors une somme de sauts
!>   dg_k = g_k - g_{k-1}   aux noeuds eta_k  (g_0 = g_{N+1} = 0),
!> et le point de collocation n'est jamais sur un noeud (valeur principale exacte) :
!>
!>   (A g)_i = 2 g_i/(a0 c_i) + 1/(4 pi h) * sum_{k=0}^{N}  dg_k / (i - k - 1/2)
!>
!> A est symétrique définie positive : gradient conjugué. Chaque produit A*x demande les sauts
!> de TOUS les noeuds : MPI_Iallgatherv, masqué par le calcul des interactions locales.
module solveur_mod
  use mpi
  use precision_mod
  use parametres_mod
  use metrologie_mod
  use decomposition_mod
  use aile_mod
  implicit none
  private
  public :: t_systeme, construire_systeme, gradient_conjugue, coefficients, residu_vrai

  type :: t_systeme
    type(t_decomp) :: dec
    real(dp) :: h = 0.0_dp, inv4pih = 0.0_dp
    logical  :: recouvrement = .true.
    real(dp), allocatable :: y(:), diag(:), rhs(:)   !< centres, 2/(a0 c), second membre (locaux)
    real(dp), allocatable :: dx_loc(:)               !< sauts aux noeuds possédés
    real(dp), allocatable :: dx_tout(:)              !< sauts aux noeuds 0:N (tampon de réception)
  end type t_systeme

contains

  subroutine construire_systeme(p, dec, sys)
    type(t_params), intent(in)    :: p
    type(t_decomp), intent(in)    :: dec
    type(t_systeme), intent(inout) :: sys
    integer :: i, ig

    sys%dec = dec
    sys%h = p%envergure/real(p%n, dp)
    sys%inv4pih = 1.0_dp/(4.0_dp*pi*sys%h)
    sys%recouvrement = (p%recouvrement == 1)
    allocate (sys%y(dec%n_loc), sys%diag(dec%n_loc), sys%rhs(dec%n_loc))
    allocate (sys%dx_loc(dec%n_noeuds_loc), sys%dx_tout(0:dec%n_glob))
    do i = 1, dec%n_loc
      ig = dec%decalage + i
      sys%y(i)    = -0.5_dp*p%envergure + (real(ig, dp) - 0.5_dp)*sys%h
      sys%diag(i) = 2.0_dp/(p%pente_a0*corde(sys%y(i), p))
      sys%rhs(i)  = incidence_utile(sys%y(i), p)
    end do
  end subroutine construire_systeme

  !> Somme des interactions  dx(j) / (ig - k - 1/2)  pour les noeuds k = k0, k0+1, ...
  pure function somme_noeuds(ig, dx, k0) result(s)
    integer, intent(in)  :: ig, k0
    real(dp), intent(in) :: dx(:)
    real(dp) :: s
    integer :: j

    s = 0.0_dp
    do j = 1, size(dx)
      s = s + dx(j)/(real(ig - (k0 + j - 1), dp) - 0.5_dp)
    end do
  end function somme_noeuds

  !> ax = A x, avec x et ax répartis par blocs de panneaux.
  subroutine produit_matrice_vecteur(sys, x, ax, chrono)
    type(t_systeme), intent(inout) :: sys
    real(dp), intent(in)           :: x(:)
    real(dp), intent(out)          :: ax(:)
    type(t_chrono), intent(inout)  :: chrono

    integer :: rq(2), rq_ag
    real(dp) :: g_gauche, g_envoi, s, t0, s_loc(size(x))
    integer  :: i, j, n_loc, ncnt, decal, nfin, nglob
    logical  :: fini
    integer :: ierr

    n_loc = sys%dec%n_loc
    ncnt  = sys%dec%n_noeuds_loc
    decal = sys%dec%decalage
    nglob = sys%dec%n_glob

    ! 1. Maille voisine : dernier panneau du rang de gauche (point-à-point non bloquant)
    t0 = MPI_Wtime()
    g_gauche = 0.0_dp                       ! reste 0 au bord de l'aile (voisin MPI_PROC_NULL)
    g_envoi  = x(n_loc)
    call MPI_Irecv(g_gauche, 1, MPI_DOUBLE_PRECISION, sys%dec%gauche, 11, sys%dec%comm, rq(1), ierr)
    call MPI_Isend(g_envoi, 1, MPI_DOUBLE_PRECISION, sys%dec%droite, 11, sys%dec%comm, rq(2), ierr)
    call MPI_Waitall(2, rq, MPI_STATUSES_IGNORE, ierr)
    chrono%halo = chrono%halo + (MPI_Wtime() - t0)

    ! 2. Sauts de circulation aux noeuds possédés (noeud decal + j - 1 = bord gauche du panneau j)
    sys%dx_loc(1) = x(1) - g_gauche
    do j = 2, n_loc
      sys%dx_loc(j) = x(j) - x(j-1)
    end do
    if (sys%dec%rang == sys%dec%nproc - 1) sys%dx_loc(n_loc + 1) = -x(n_loc)

    ! 3. Rassemblement de tous les sauts : non bloquant (avec recouvrement) ou bloquant
    fini  = .not. sys%recouvrement
    rq_ag = MPI_REQUEST_NULL
    if (sys%recouvrement) then
      call MPI_Iallgatherv(sys%dx_loc, ncnt, MPI_DOUBLE_PRECISION, sys%dx_tout, &
                           sys%dec%comptes, sys%dec%deplacements, MPI_DOUBLE_PRECISION, &
                           sys%dec%comm, rq_ag, ierr)
    else
      t0 = MPI_Wtime()
      call MPI_Allgatherv(sys%dx_loc, ncnt, MPI_DOUBLE_PRECISION, sys%dx_tout, &
                          sys%dec%comptes, sys%dec%deplacements, MPI_DOUBLE_PRECISION, &
                          sys%dec%comm, ierr)
      chrono%attente = chrono%attente + (MPI_Wtime() - t0)
    end if

    ! 4. Noyau autonome : interactions avec les noeuds locaux, pendant le transfert.
    !    (le tampon de réception sys%dx_tout n'est pas touché tant que la requête est ouverte)
    t0 = MPI_Wtime()
    do i = 1, n_loc
      s_loc(i) = somme_noeuds(decal + i, sys%dx_loc, decal)
      if (.not. fini .and. mod(i, 16) == 0) &        ! fait progresser la communication asynchrone
        call MPI_Test(rq_ag, fini, MPI_STATUS_IGNORE, ierr)
    end do
    chrono%calcul_local = chrono%calcul_local + (MPI_Wtime() - t0)

    ! 5. Validation de la réception complète
    if (sys%recouvrement) then
      t0 = MPI_Wtime()
      call MPI_Wait(rq_ag, MPI_STATUS_IGNORE, ierr)
      chrono%attente = chrono%attente + (MPI_Wtime() - t0)
    end if

    ! 6. Interactions avec les noeuds des autres rangs
    t0 = MPI_Wtime()
    nfin = decal + ncnt
    do i = 1, n_loc
      s = 0.0_dp
      if (decal > 0)     s = s + somme_noeuds(decal + i, sys%dx_tout(0:decal-1), 0)
      if (nfin <= nglob) s = s + somme_noeuds(decal + i, sys%dx_tout(nfin:nglob), nfin)
      ax(i) = sys%diag(i)*x(i) + sys%inv4pih*(s_loc(i) + s)
    end do
    chrono%calcul_distant = chrono%calcul_distant + (MPI_Wtime() - t0)
  end subroutine produit_matrice_vecteur

  function produit_scalaire(a, b, comm, chrono) result(s)
    real(dp), intent(in)          :: a(:), b(:)
    integer, intent(in)           :: comm
    type(t_chrono), intent(inout) :: chrono
    real(dp) :: s, t0
    integer :: ierr

    s = dot_product(a, b)
    t0 = MPI_Wtime()
    call MPI_Allreduce(MPI_IN_PLACE, s, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
    chrono%reductions = chrono%reductions + (MPI_Wtime() - t0)
  end function produit_scalaire

  !> Gradient conjugué distribué pour A g = rhs (départ à g = 0).
  subroutine gradient_conjugue(sys, g, tol, itmax, nit, residu_rel, chrono)
    type(t_systeme), intent(inout) :: sys
    real(dp), intent(out)          :: g(:)
    real(dp), intent(in)           :: tol
    integer, intent(in)            :: itmax
    integer, intent(out)           :: nit
    real(dp), intent(out)          :: residu_rel
    type(t_chrono), intent(inout)  :: chrono

    real(dp), allocatable :: r(:), d(:), ad(:)
    real(dp) :: rs, rs0, rs_new, alpha

    allocate (r(size(g)), d(size(g)), ad(size(g)))
    g = 0.0_dp
    r = sys%rhs
    d = r
    rs = produit_scalaire(r, r, sys%dec%comm, chrono)
    rs0 = rs
    nit = 0
    residu_rel = 0.0_dp
    if (rs0 <= tiny(1.0_dp)) return

    do while (nit < itmax)
      nit = nit + 1
      call produit_matrice_vecteur(sys, d, ad, chrono)
      alpha = rs/produit_scalaire(d, ad, sys%dec%comm, chrono)
      g = g + alpha*d
      r = r - alpha*ad
      rs_new = produit_scalaire(r, r, sys%dec%comm, chrono)
      residu_rel = sqrt(rs_new/rs0)
      if (residu_rel < tol) exit
      d = r + (rs_new/rs)*d
      rs = rs_new
    end do
  end subroutine gradient_conjugue

  !> Residu relatif vrai ||rhs - A g|| / ||rhs|| (hors chronométrage).
  function residu_vrai(sys, g) result(res)
    type(t_systeme), intent(inout) :: sys
    real(dp), intent(in) :: g(:)
    real(dp) :: res, norme_rhs
    real(dp), allocatable :: r(:)
    type(t_chrono) :: poubelle

    allocate (r(size(g)))
    call produit_matrice_vecteur(sys, g, r, poubelle)
    r = sys%rhs - r
    norme_rhs = produit_scalaire(sys%rhs, sys%rhs, sys%dec%comm, poubelle)
    if (norme_rhs <= tiny(1.0_dp)) then
      res = 0.0_dp
    else
      res = sqrt(produit_scalaire(r, r, sys%dec%comm, poubelle) / norme_rhs)
    end if
  end function residu_vrai

  !> Coefficients globaux : CL = 2 sum(g) h / S ;  CDi = 2 sum(g alpha_i) h / S
  !> avec alpha_i = rhs - 2 g/(a0 c) (angle induit, déduit de l'équation).
  subroutine coefficients(sys, p, g, cl, cdi, ar)
    type(t_systeme), intent(in) :: sys
    type(t_params), intent(in)  :: p
    real(dp), intent(in)        :: g(:)
    real(dp), intent(out)       :: cl, cdi, ar
    real(dp) :: somme(2), s_ref
    integer :: ierr

    somme(1) = sum(g)*sys%h
    somme(2) = sum(g*(sys%rhs - sys%diag*g))*sys%h
    call MPI_Allreduce(MPI_IN_PLACE, somme, 2, MPI_DOUBLE_PRECISION, MPI_SUM, sys%dec%comm, ierr)
    s_ref = surface_alaire(p)
    ar  = p%envergure**2/s_ref
    cl  = 2.0_dp*somme(1)/s_ref
    cdi = 2.0_dp*somme(2)/s_ref
  end subroutine coefficients

end module solveur_mod
