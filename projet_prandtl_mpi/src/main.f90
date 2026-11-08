!> Projet 2 (MSN) : équation de Prandtl en parallèle (Fortran 2008, mpi_f08).
!> Topologie cartésienne 1D, MPI_Iallgatherv avec recouvrement, métrologie MPI_Wtime.
program prandtl_mpi
  use mpi
  use precision_mod
  use parametres_mod
  use metrologie_mod
  use decomposition_mod
  use aile_mod
  use solveur_mod
  implicit none

  character(len=*), parameter :: entete = 'P,N,recouvrement,iterations,t_min,t_med,t_attente,'// &
    't_calcul_local,t_calcul_distant,t_halo,t_reductions,CL,CDi,residu,AR'

  type(t_params)  :: p
  type(t_decomp)  :: dec
  type(t_systeme) :: sys
  type(t_chrono)  :: chrono, chrono_max
  real(dp), allocatable :: g(:), t_loc(:), t_max(:)
  real(dp) :: t0, cl, cdi, ar, residu, residu_verif, t_min, t_med, s_ref, q_dyn
  integer  :: irep, nit, ierr

  call MPI_Init(ierr)
  call lire_arguments(p)
  call creer_topologie(p%n, dec)
  call construire_systeme(p, dec, sys)
  allocate (g(dec%n_loc), t_loc(p%nrep), t_max(p%nrep))

  ! Résolution de chauffe, non chronométrée (allocations, pages mémoire, connexions réseau)
  chrono = t_chrono()
  call gradient_conjugue(sys, g, p%tol, p%itmax, nit, residu, chrono)

  ! Mesures : barrière avant, barrière après (capture du dernier arrivé), pire temps par MPI_MAX
  do irep = 1, p%nrep
    chrono = t_chrono()
    call MPI_Barrier(dec%comm, ierr)
    t0 = MPI_Wtime()
    call gradient_conjugue(sys, g, p%tol, p%itmax, nit, residu, chrono)
    call MPI_Barrier(dec%comm, ierr)
    t_loc(irep) = MPI_Wtime() - t0
  end do
  call MPI_Reduce(t_loc, t_max, p%nrep, MPI_DOUBLE_PRECISION, MPI_MAX, 0, dec%comm, ierr)
  call reduire_chrono_max(chrono, dec%comm, chrono_max)

  call coefficients(sys, p, g, cl, cdi, ar)
  residu_verif = residu_vrai(sys, g)
  if (p%dump == 1) call ecrire_profil()

  if (dec%rang == 0) then
    t_min = minval(t_max)
    t_med = mediane(t_max)
    s_ref = surface_alaire(p)
    q_dyn = 0.5_dp*p%rho*p%vinf**2
    call afficher_bilan()
    call ecrire_resultats()
  end if

  call MPI_Finalize(ierr)

contains

  subroutine afficher_bilan()
    real(dp) :: cl_th, cdi_th, e_env
    character(len=1) :: mode
    integer :: r

    mode = 'N'
    if (p%recouvrement == 1) mode = 'O'
    write (*, '(a)') '=================== Prandtl / MPI ==================='
    write (*, '(a,i0,a,i0,a,i0,a)') ' Processus P = ', dec%nproc, '   panneaux N = ', p%n, &
      '   (', p%n/dec%nproc, ' par processus)'
    write (*, '(a)', advance='no') ' Sections réellement distribuées :'
    do r = 0, dec%nproc - 1
      write (*, '(a,i0,a,i0,a,i0)', advance='no') ' rang ', r, ' -> ', &
        dec%deplacements(r) + 1, '..', dec%deplacements(r) + dec%n_pan(r)
    end do
    write (*, *)
    write (*, '(a,a)') ' Iallgatherv + recouvrement : ', merge('oui', 'non', mode == 'O')
    write (*, '(a,i0,a,es9.2,a,es9.2)') ' Gradient conjugue : ', nit, ' iterations, residu ', &
      residu, ', residu vrai ', residu_verif
    write (*, '(a,f8.3,a,f9.3,a,f7.3)') ' Allongement lambda = ', ar, '   S = ', s_ref, ' m2'
    write (*, '(a,f10.6,a,f10.6)') ' CL  = ', cl, '    CDi = ', cdi
    if (cdi > 0.0_dp) write (*, '(a,f8.4)') ' Efficacite d''envergure e = CL^2/(pi lambda CDi) = ', &
      cl**2/(pi*ar*cdi)
    write (*, '(a,f10.3,a,f9.4,a)') ' Portance = ', q_dyn*s_ref*cl, ' N,  traînée induite = ', &
      q_dyn*s_ref*cdi, ' N'
    if (p%elliptique == 1 .and. abs(p%vrillage_deg) <= tiny(1.0_dp)) then
      cl_th  = p%pente_a0*(p%incidence_deg - p%alpha_l0_deg)*pi/180.0_dp/(1.0_dp + p%pente_a0/(pi*ar))
      cdi_th = cl_th**2/(pi*ar)
      e_env = 100.0_dp*(cl/cl_th - 1.0_dp)
      write (*, '(a,f10.6,a,f10.6,a,f8.4,a)') ' Theorie (aile elliptique) : CL = ', cl_th, &
        '  CDi = ', cdi_th, '   ecart CL ', e_env, ' %'
    end if
    write (*, '(a,f10.4,a,f10.4,a,i0,a)') ' Temps (pire des rangs) : min ', t_min, ' s, mediane ', &
      t_med, ' s sur ', p%nrep, ' repetitions'
    write (*, '(a)') ' Pire temps par phase (derniere repetition, s) :'
    write (*, '(a,f9.4,a,f9.4)') '   calcul local   ', chrono_max%calcul_local, &
      '   calcul distant ', chrono_max%calcul_distant
    write (*, '(a,f9.4,a,f9.4)') '   attente (Wait) ', chrono_max%attente, &
      '   halo           ', chrono_max%halo
    write (*, '(a,f9.4)') '   reductions     ', chrono_max%reductions
    write (*, '(a)') '====================================================='
  end subroutine afficher_bilan

  !> Ligne CSV sur la sortie standard (reperee par "CSV,") et ajout au fichier de resultats.
  subroutine ecrire_resultats()
    character(len=600) :: ligne
    real(dp) :: valeurs(11)
    logical :: existe
    integer :: u, k

    valeurs = [t_min, t_med, chrono_max%attente, chrono_max%calcul_local, chrono_max%calcul_distant, &
               chrono_max%halo, chrono_max%reductions, cl, cdi, residu_verif, ar]
    write (ligne, '(i0,3(",",i0))') dec%nproc, p%n, p%recouvrement, nit
    do k = 1, size(valeurs)
      ligne = trim(ligne)//','//reel_texte(valeurs(k))
    end do
    write (*, '(a)') 'CSV,'//trim(ligne)
    inquire (file=trim(p%sortie), exist=existe)
    open (newunit=u, file=trim(p%sortie), status='unknown', position='append', action='write')
    if (.not. existe) write (u, '(a)') entete
    write (u, '(a)') trim(ligne)
    close (u)
  end subroutine ecrire_resultats

  !> Réel en notation scientifique, sans blancs (pour les fichiers CSV).
  function reel_texte(x) result(t)
    real(dp), intent(in) :: x
    character(len=:), allocatable :: t
    character(len=24) :: tampon

    write (tampon, '(es16.8e3)') x
    t = trim(adjustl(tampon))
  end function reel_texte

  !> Profil d'envergure (y, c, Gamma/V, alpha_i) rassemblé sur le rang 0.
  subroutine ecrire_profil()
    real(dp), allocatable :: g_tout(:), a_tout(:), a_loc(:)
    real(dp) :: yy, dg, g_gauche, g_droite
    integer :: i, u, ug, rang_proprietaire

    allocate (g_tout(p%n), a_tout(p%n), a_loc(dec%n_loc))
    a_loc = sys%rhs - sys%diag*g
    call MPI_Gatherv(g, dec%n_loc, MPI_DOUBLE_PRECISION, g_tout, dec%n_pan, dec%deplacements, &
                     MPI_DOUBLE_PRECISION, 0, dec%comm, ierr)
    call MPI_Gatherv(a_loc, dec%n_loc, MPI_DOUBLE_PRECISION, a_tout, dec%n_pan, dec%deplacements, &
                     MPI_DOUBLE_PRECISION, 0, dec%comm, ierr)
    if (dec%rang == 0) then
      open (newunit=u, file='profil_gamma.csv', status='replace', action='write')
      write (u, '(a)') 'y_m,corde_m,gamma_sur_V_m,alpha_induit_rad'
      do i = 1, p%n
        yy = -0.5_dp*p%envergure + (real(i, dp) - 0.5_dp)*sys%h
        write (u, '(a)') reel_texte(yy)//','//reel_texte(corde(yy, p))//','//reel_texte(g_tout(i)) &
                         //','//reel_texte(a_tout(i))
      end do
      close (u)

      open (newunit=ug, file='profil_gradient.csv', status='replace', action='write')
      write (ug, '(a)') 'noeud,y_noeud_m,gradient_saut_gamma_sur_V,gradient_gamma_sur_V_par_m,'// &
                        'gamma_gauche_sur_V,gamma_droite_sur_V,rang_proprietaire'
      do i = 0, p%n
        if (i == 0) then
          dg = g_tout(1)
        else if (i == p%n) then
          dg = -g_tout(p%n)
        else
          dg = g_tout(i+1) - g_tout(i)
        end if
        g_gauche = 0.0_dp
        g_droite = 0.0_dp
        if (i > 0) g_gauche = g_tout(i)
        if (i < p%n) g_droite = g_tout(i+1)
        if (i == 0) then
          rang_proprietaire = 0
        else
          rang_proprietaire = 0
          do while (rang_proprietaire < dec%nproc - 1 .and. &
                    i >= dec%deplacements(rang_proprietaire) + dec%n_pan(rang_proprietaire))
            rang_proprietaire = rang_proprietaire + 1
          end do
        end if
        yy = -0.5_dp*p%envergure + real(i, dp)*sys%h
        write (ug, '(i0,a)') i, ','//reel_texte(yy)//','//reel_texte(dg)//','// &
          reel_texte(dg/sys%h)//','//reel_texte(g_gauche)//','// &
          reel_texte(g_droite)//','//reel_texte(real(rang_proprietaire, dp))
      end do
      close (ug)
    end if
  end subroutine ecrire_profil

end program prandtl_mpi
