!> Outils de métrologie : cumul des temps par phase, réduction du pire temps (MPI_MAX), médiane.
module metrologie_mod
  use mpi
  use precision_mod
  implicit none
  private
  public :: t_chrono, reduire_chrono_max, mediane

  !> Temps cumulés (secondes) par phase, mesurés avec MPI_Wtime.
  type :: t_chrono
    real(dp) :: calcul_local   = 0.0_dp   !< noyau autonome, exécuté pendant le transfert
    real(dp) :: calcul_distant = 0.0_dp   !< contributions des noeuds reçus des autres rangs
    real(dp) :: attente        = 0.0_dp   !< MPI_Wait / MPI_Allgatherv : latence non masquée
    real(dp) :: halo           = 0.0_dp   !< échange de la maille voisine (Isend/Irecv)
    real(dp) :: reductions     = 0.0_dp   !< MPI_Allreduce des produits scalaires
  end type t_chrono

contains

  !> Pire valeur de chaque phase sur tous les rangs (résultat significatif sur le rang 0).
  subroutine reduire_chrono_max(c, comm, c_max)
    type(t_chrono), intent(in)  :: c
    integer, intent(in)          :: comm
    type(t_chrono), intent(out) :: c_max
    real(dp) :: v(5), vmax(5)
    integer :: ierr

    v = [c%calcul_local, c%calcul_distant, c%attente, c%halo, c%reductions]
    vmax = 0.0_dp
    call MPI_Reduce(v, vmax, 5, MPI_DOUBLE_PRECISION, MPI_MAX, 0, comm, ierr)
    c_max = t_chrono(vmax(1), vmax(2), vmax(3), vmax(4), vmax(5))
  end subroutine reduire_chrono_max

  function mediane(v) result(m)
    real(dp), intent(in) :: v(:)
    real(dp) :: m
    real(dp) :: w(size(v)), tmp
    integer :: i, j, n

    n = size(v)
    w = v
    do i = 2, n                       ! tri par insertion (n est petit)
      tmp = w(i)
      j = i - 1
      do while (j >= 1)
        if (w(j) <= tmp) exit
        w(j+1) = w(j)
        j = j - 1
      end do
      w(j+1) = tmp
    end do
    if (mod(n, 2) == 1) then
      m = w((n+1)/2)
    else
      m = 0.5_dp*(w(n/2) + w(n/2+1))
    end if
  end function mediane

end module metrologie_mod
