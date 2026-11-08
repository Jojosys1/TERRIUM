!> Paramètres du cas, lus sur la ligne de commande sous la forme cle=valeur.
!> Exemple : mpirun -np 4 ./bin/prandtl n=4096 incidence=5 rep=3
module parametres_mod
  use mpi
  use precision_mod
  implicit none
  private
  public :: t_params, lire_arguments

  type :: t_params
    integer  :: n = 1024                      !< nombre de panneaux sur l'envergure (pair)
    real(dp) :: envergure = 2.0_dp            !< b (m)
    real(dp) :: corde_racine = 0.35_dp        !< c a la racine (m)
    real(dp) :: corde_saumon = 0.18_dp        !< c au saumon (m), ignoree si elliptique
    real(dp) :: incidence_deg = 5.0_dp        !< incidence geometrique (deg)
    real(dp) :: vrillage_deg = -2.0_dp        !< vrillage au saumon (deg), lineaire en |y|
    real(dp) :: alpha_l0_deg = 0.0_dp         !< incidence de portance nulle du profil (deg)
    real(dp) :: pente_a0 = 2.0_dp*pi          !< pente de portance 2D (1/rad)
    integer  :: elliptique = 0                !< 1 : corde elliptique (cas de validation)
    real(dp) :: vinf = 15.0_dp                !< vitesse (m/s), pour les efforts dimensionnels
    real(dp) :: rho = 1.225_dp                !< masse volumique (kg/m3)
    real(dp) :: tol = 1.0e-10_dp              !< tolerance relative du gradient conjugue
    integer  :: itmax = 5000                  !< iterations max
    integer  :: nrep = 3                      !< repetitions chronometrees
    integer  :: recouvrement = 1              !< 1 : MPI_Iallgatherv + recouvrement, 0 : MPI_Allgatherv
    integer  :: dump = 0                      !< 1 : ecrit profil_gamma.csv
    character(len=256) :: sortie = 'resultats.csv'
  end type t_params

contains

  subroutine lire_arguments(p)
    type(t_params), intent(inout) :: p
    integer :: i, pos, ios
    character(len=256) :: arg, cle, val

    do i = 1, command_argument_count()
      call get_command_argument(i, arg)
      pos = index(arg, '=')
      if (pos < 2) call abandon('argument invalide (attendu cle=valeur) : '//trim(arg))
      cle = arg(1:pos-1)
      val = arg(pos+1:)
      ios = 0
      select case (trim(cle))
      case ('n');            read (val, *, iostat=ios) p%n
      case ('envergure');    read (val, *, iostat=ios) p%envergure
      case ('croot');        read (val, *, iostat=ios) p%corde_racine
      case ('ctip');         read (val, *, iostat=ios) p%corde_saumon
      case ('incidence');    read (val, *, iostat=ios) p%incidence_deg
      case ('vrillage');     read (val, *, iostat=ios) p%vrillage_deg
      case ('alpha_l0');     read (val, *, iostat=ios) p%alpha_l0_deg
      case ('a0');           read (val, *, iostat=ios) p%pente_a0
      case ('elliptique');   read (val, *, iostat=ios) p%elliptique
      case ('vinf');         read (val, *, iostat=ios) p%vinf
      case ('rho');          read (val, *, iostat=ios) p%rho
      case ('tol');          read (val, *, iostat=ios) p%tol
      case ('itmax');        read (val, *, iostat=ios) p%itmax
      case ('rep');          read (val, *, iostat=ios) p%nrep
      case ('recouvrement'); read (val, *, iostat=ios) p%recouvrement
      case ('dump');         read (val, *, iostat=ios) p%dump
      case ('sortie');       p%sortie = val
      case default;          call abandon('cle inconnue : '//trim(cle))
      end select
      if (ios /= 0) call abandon('valeur invalide pour : '//trim(cle))
    end do

    if (mod(p%n, 2) /= 0) p%n = p%n + 1      ! n pair : un noeud en y = 0
    if (p%n < 2) call abandon('n doit etre >= 2')
    if (p%envergure <= 0.0_dp .or. p%corde_racine <= 0.0_dp .or. p%corde_saumon <= 0.0_dp) &
      call abandon('envergure et cordes doivent etre positives')
    if (p%pente_a0 <= 0.0_dp .or. p%tol <= 0.0_dp) call abandon('a0 et tol doivent etre positifs')
    if (p%nrep < 1 .or. p%itmax < 1) call abandon('rep et itmax doivent etre >= 1')
  end subroutine lire_arguments

  subroutine abandon(message)
    character(len=*), intent(in) :: message
    integer :: ierr
    write (*, '(a)') 'ERREUR : '//message
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end subroutine abandon

end module parametres_mod
