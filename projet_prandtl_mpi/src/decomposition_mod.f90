!> Topologie cartésienne 1D et décomposition de l'envergure en blocs contigus.
!>
!> Panneaux globaux 1..N, noeuds globaux 0..N (le panneau i est limité par les noeuds i-1 et i).
!> Chaque rang possède les panneaux  decalage+1 .. decalage+n_loc  et les noeuds
!> decalage .. decalage+n_loc-1 (noeuds de gauche de ses panneaux) ; le dernier rang
!> possède en plus le noeud N.
module decomposition_mod
  use mpi
  implicit none
  private
  public :: t_decomp, creer_topologie

  type :: t_decomp
    integer :: comm                            !< communicateur cartésien 1D
    integer :: rang = 0, nproc = 1
    integer :: gauche = 0, droite = 0         !< voisins (MPI_PROC_NULL aux extrémités)
    integer :: n_glob = 0, n_loc = 0, decalage = 0, n_noeuds_loc = 0
    integer, allocatable :: n_pan(:)          !< panneaux par rang        (0:nproc-1)
    integer, allocatable :: comptes(:)        !< noeuds par rang          (0:nproc-1)
    integer, allocatable :: deplacements(:)   !< premier noeud de chaque rang (0:nproc-1)
  end type t_decomp

contains

  subroutine creer_topologie(n_glob, dec)
    integer, intent(in)         :: n_glob
    type(t_decomp), intent(out) :: dec
    integer :: dims(1), r, base, reste, ierr
    logical :: periodes(1)

    call MPI_Comm_size(MPI_COMM_WORLD, dec%nproc, ierr)
    if (n_glob < dec%nproc) then
      write (*, '(a)') 'ERREUR : il faut au moins un panneau par processus (n >= P)'
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if

    dims(1) = 0
    periodes(1) = .false.                                   ! conditions aux limites : pas de périodicité
    call MPI_Dims_create(dec%nproc, 1, dims, ierr)
    call MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periodes, .false., dec%comm, ierr)
    call MPI_Comm_rank(dec%comm, dec%rang, ierr)
    call MPI_Cart_shift(dec%comm, 0, 1, dec%gauche, dec%droite, ierr)

    allocate (dec%n_pan(0:dec%nproc-1), dec%comptes(0:dec%nproc-1), dec%deplacements(0:dec%nproc-1))
    base  = n_glob/dec%nproc
    reste = mod(n_glob, dec%nproc)
    do r = 0, dec%nproc-1
      dec%n_pan(r) = base
      if (r < reste) dec%n_pan(r) = base + 1
      dec%comptes(r) = dec%n_pan(r)
      if (r == 0) then
        dec%deplacements(r) = 0
      else
        dec%deplacements(r) = dec%deplacements(r-1) + dec%n_pan(r-1)
      end if
    end do
    dec%comptes(dec%nproc-1) = dec%comptes(dec%nproc-1) + 1   ! dernier noeud (y = +b/2)

    dec%n_glob       = n_glob
    dec%n_loc        = dec%n_pan(dec%rang)
    dec%decalage     = dec%deplacements(dec%rang)
    dec%n_noeuds_loc = dec%comptes(dec%rang)
  end subroutine creer_topologie

end module decomposition_mod
