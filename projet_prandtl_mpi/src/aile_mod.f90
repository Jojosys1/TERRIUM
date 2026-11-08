!> Géométrie de l'aile pour la ligne portante de Prandtl (envergure y dans [-b/2, b/2]).
module aile_mod
  use precision_mod
  use parametres_mod
  implicit none
  private
  public :: corde, incidence_utile, surface_alaire

contains

  !> Corde locale c(y) : trapézoïdale (racine -> saumon) ou elliptique.
  pure function corde(y, p) result(c)
    real(dp), intent(in) :: y
    type(t_params), intent(in) :: p
    real(dp) :: c, eta

    eta = abs(y)/(0.5_dp*p%envergure)
    if (p%elliptique == 1) then
      c = p%corde_racine*sqrt(max(1.0_dp - eta**2, 0.0_dp))
    else
      c = p%corde_racine + (p%corde_saumon - p%corde_racine)*eta
    end if
  end function corde

  !> Second membre : incidence géométrique + vrillage - incidence de portance nulle (radians).
  pure function incidence_utile(y, p) result(a)
    real(dp), intent(in) :: y
    type(t_params), intent(in) :: p
    real(dp) :: a

    a = (p%incidence_deg + p%vrillage_deg*abs(y)/(0.5_dp*p%envergure) - p%alpha_l0_deg) &
        *pi/180.0_dp
  end function incidence_utile

  !> Surface de référence S = intégrale de c(y) dy (valeur exacte).
  pure function surface_alaire(p) result(s)
    type(t_params), intent(in) :: p
    real(dp) :: s

    if (p%elliptique == 1) then
      s = 0.25_dp*pi*p%corde_racine*p%envergure
    else
      s = 0.5_dp*p%envergure*(p%corde_racine + p%corde_saumon)
    end if
  end function surface_alaire

end module aile_mod
