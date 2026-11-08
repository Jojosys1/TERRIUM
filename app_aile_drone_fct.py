"""
Aile de drone : géométrie paramétrique, maillage surfacique et écoulement potentiel (VLM).

Lancer :  streamlit run app_aile_drone_vlm.py
Dépendances : streamlit, numpy, pandas, plotly

Repère lié à l'aile : x vers le bord de fuite, y en envergure, z vers le haut.
L'aile avance vers -x ; l'écoulement relatif U_inf est donc dirigé vers +x.
"""
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fortran_bridge import (
    FortranBridgeError,
    analytical_reference,
    compile_solver,
    run_solver,
    tool_status,
)

st.set_page_config(page_title="Terrium — aile de drone", layout="wide")
st.markdown("""
<div style="text-align:center; margin:0 0 1.2rem 0;">
  <svg width="76" height="76" viewBox="0 0 100 100" role="img"
       aria-label="Logo Terrium" style="display:block;margin:auto;">
    <defs>
      <linearGradient id="terriumMetal" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#2f1203"/>
        <stop offset="0.45" stop-color="#7b3d0b"/>
        <stop offset="0.7" stop-color="#4a2106"/>
        <stop offset="1" stop-color="#8b4a13"/>
      </linearGradient>
    </defs>
    <path d="M12 18 L70 18 L70 31 L29 31 L29 69 L70 69 L70 82 L12 82 Z"
          fill="url(#terriumMetal)"/>
    <path d="M30 31 L82 18 L88 24 L88 76 L30 69 L30 56 L75 64 L75 31 L30 44 Z"
          fill="url(#terriumMetal)"/>
  </svg>
  <div style="font-size:2rem;font-weight:700;letter-spacing:.18em;color:#5a2a05;">
    TERRIUM
  </div>
</div>
""", unsafe_allow_html=True)
st.title("Modélisation, maillage et écoulement autour d'une aile de drone")
st.caption("Géométrie paramétrique 3D — maillage surfacique — écoulement potentiel par méthode "
           "des réseaux de tourbillons (VLM)")

# Compatibilité entre versions de Streamlit (use_container_width -> width="stretch")
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
           else {"use_container_width": True})

CORE_EXACT = 1e-6   # m : évite 0/0 sur l'axe d'un filament (résolution du système)


# ======================================================================
# 1. GÉOMÉTRIE ET MAILLAGE SURFACIQUE
# ======================================================================
def naca4_coordinates(x_over_c, code="0012"):
    code = str(code).zfill(4)
    if len(code) != 4 or not code.isdigit():
        raise ValueError("Code NACA invalide : saisir quatre chiffres, ex. 0012.")
    m, p, t = int(code[0])/100, int(code[1])/10, int(code[2:])/100
    x = np.asarray(x_over_c, dtype=float)
    yt = 5*t*(0.2969*np.sqrt(np.maximum(x, 0))-0.1260*x
              - 0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yc = np.zeros_like(x)
    if m > 0 and p > 0:
        left = x < p
        right = ~left
        yc[left] = m/p**2*(2*p*x[left]-x[left]**2)
        yc[right] = m/(1-p)**2*((1-2*p)+2*p*x[right]-x[right]**2)
    return yc, yt


def section_points(y_abs, s, geom):
    """Coordonnées (x, z) de la ligne de cambrure, de l'extrados et de l'intrados
    à l'abscisse d'envergure |y| et à la fraction de corde s = x/c (tableaux broadcastables)."""
    span, root_chord, tip_chord, sweep_deg, dihedral_deg, twist_deg, naca = geom
    eta = y_abs/(span/2)
    chord = root_chord+(tip_chord-root_chord)*eta
    x_le = y_abs*np.tan(np.deg2rad(sweep_deg))
    z_ref = y_abs*np.tan(np.deg2rad(dihedral_deg))
    tw = np.deg2rad(twist_deg)*eta
    yc, yt = naca4_coordinates(s, naca)
    x_local = s*chord
    out = {}
    for name, zp in (("camber", yc*chord), ("upper", (yc+yt)*chord), ("lower", (yc-yt)*chord)):
        out[name] = (x_le+x_local*np.cos(tw)+zp*np.sin(tw),
                     z_ref-x_local*np.sin(tw)+zp*np.cos(tw))
    return out


def build_wing_surface(geom, n_span, n_chord):
    span, root_chord, tip_chord = geom[0], geom[1], geom[2]
    if min(span, root_chord, tip_chord) <= 0:
        raise ValueError("Les dimensions doivent être positives.")
    y = np.linspace(0, span/2, n_span+1)
    xc = np.linspace(0, 1, n_chord+1)
    Y, S = np.meshgrid(y, xc, indexing="ij")
    sec = section_points(Y, S, geom)
    result = {}
    for name, key in (("Extrados", "upper"), ("Intrados", "lower")):
        X, Z = sec[key]
        result[name] = (
            np.concatenate([X[:, ::-1], X[:, 1:]], axis=1),
            np.concatenate([-Y[:, ::-1], Y[:, 1:]], axis=1),
            np.concatenate([Z[:, ::-1], Z[:, 1:]], axis=1),
        )
    return result


def mesh_from_surfaces(surfaces, element_type):
    node_blocks, elements = [], []
    offset = 0
    for X, Y, Z in surfaces.values():
        nr, nc = X.shape
        node_blocks.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        for i in range(nr-1):
            for j in range(nc-1):
                a, b = offset+i*nc+j, offset+i*nc+j+1
                c, d = offset+(i+1)*nc+j+1, offset+(i+1)*nc+j
                if element_type == "Quadrilatères":
                    elements.append([a, b, c, d])
                else:
                    elements.extend([[a, b, c], [a, c, d]])
        offset += nr*nc
    return np.vstack(node_blocks), np.asarray(elements, dtype=int)


def make_figure(surfaces, with_mesh, title):
    fig = go.Figure()
    for name, (X, Y, Z) in surfaces.items():
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, name=name, showscale=False, opacity=0.96,
            colorscale="Blues" if name == "Extrados" else "Greys"))
        if with_mesh:
            lx, ly, lz = [], [], []
            nr, nc = X.shape
            for i in range(nr):
                lx += X[i, :].tolist()+[None]
                ly += Y[i, :].tolist()+[None]
                lz += Z[i, :].tolist()+[None]
            for j in range(nc):
                lx += X[:, j].tolist()+[None]
                ly += Y[:, j].tolist()+[None]
                lz += Z[:, j].tolist()+[None]
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line=dict(color="#1565C0" if name == "Extrados" else "#6A1B9A", width=2),
                showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        title=title, height=560, margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(xaxis_title="x — corde (m)", yaxis_title="y — envergure (m)",
                   zaxis_title="z — hauteur (m)", aspectmode="data"))
    return fig


# ======================================================================
# 2. MODÈLE AÉRODYNAMIQUE : VLM (écoulement potentiel incompressible)
#    u = grad(phi), div u = 0, rot u = 0, Delta phi = 0
#    Aile = nappe tourbillonnaire sur la surface moyenne (cambrure), épaisseur ignorée.
# ======================================================================
def camber_surface(y, s, geom):
    """Point (x, y, z) de la surface moyenne pour y (signé) et s = x/c."""
    y, s = np.broadcast_arrays(np.asarray(y, float), np.asarray(s, float))
    X, Z = section_points(np.abs(y), s, geom)["camber"]
    return np.stack([X, y, Z], axis=-1)


def build_panels(geom, nx, ny):
    """Panneaux VLM : segment porteur au quart de corde (A->B, y croissant),
    point de contrôle aux 3/4 de corde, normale exacte de la surface moyenne."""
    y = -(geom[0]/2)*np.cos(np.pi*np.arange(ny+1)/ny)        # espacement en cosinus
    s = 0.5*(1-np.cos(np.pi*np.arange(nx+1)/nx))
    P = camber_surface(y[None, :], s[:, None], geom)         # noeuds (nx+1, ny+1, 3)
    ds = np.diff(s)[:, None]
    s_q, s_c = s[:-1, None]+0.25*ds, s[:-1, None]+0.75*ds
    y_m = (0.5*(y[:-1]+y[1:]))[None, :]
    A = camber_surface(y[None, :-1], s_q, geom)
    B = camber_surface(y[None, 1:], s_q, geom)
    C = camber_surface(y_m, s_c, geom)
    e = 1e-6                                                 # tangentes par différences centrées
    t_s = camber_surface(y_m, s_c+e, geom)-camber_surface(y_m, s_c-e, geom)
    t_y = camber_surface(y_m+e, s_c, geom)-camber_surface(y_m-e, s_c, geom)
    n = np.cross(t_s, t_y)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    d1, d2 = P[1:, 1:]-P[:-1, :-1], P[:-1, 1:]-P[1:, :-1]
    S_ref = 0.5*np.cross(d1, d2)[..., 2].sum()               # surface projetée dans le plan xy
    flat = lambda a: a.reshape(-1, 3)
    return flat(A), flat(B), flat(C), flat(n), S_ref, P


def freestream(airspeed, wind_speed, wind_dir_deg, aoa_deg):
    """Écoulement relatif dans le repère de l'aile.
    Vent : direction vers laquelle il souffle, angle psi dans le plan xy (0° = vent de face)."""
    psi, alpha = np.deg2rad(wind_dir_deg), np.deg2rad(aoa_deg)
    ux, uy = airspeed+wind_speed*np.cos(psi), wind_speed*np.sin(psi)
    if ux <= 0.5:
        raise ValueError("Le vent arrière compense la vitesse de l'aile : écoulement relatif "
                         "inversé ou quasi nul, hors du domaine du modèle.")
    V = float(np.hypot(ux, uy))
    beta = np.arctan2(uy, ux)                                # dérapage
    d = np.array([np.cos(alpha)*np.cos(beta), np.sin(beta), np.sin(alpha)*np.cos(beta)])
    return V, d, float(np.rad2deg(beta))


def _segment_fini(P, A, B, core):
    """Biot-Savart, segment AB, circulation unitaire de A vers B. P:(M,3) A,B:(N,3) -> (M,N,3)."""
    r1, r2, r0 = P[:, None, :]-A[None], P[:, None, :]-B[None], (B-A)[None]
    cr = np.cross(r1, r2)
    n1 = np.maximum(np.linalg.norm(r1, axis=-1), 1e-12)
    n2 = np.maximum(np.linalg.norm(r2, axis=-1), 1e-12)
    den = np.sum(cr**2, -1)+core**2*np.sum(r0**2, -1)        # noyau de Scully (régularisation)
    k = np.sum(r0*(r1/n1[..., None]-r2/n2[..., None]), -1)/den/(4*np.pi)
    return cr*k[..., None]


def _semi_infini(P, A, d, core):
    """Filament semi-infini issu de A, dirigé selon d (unitaire), circulation unitaire."""
    r1 = P[:, None, :]-A[None]
    cr = np.cross(d, r1)
    n1 = np.maximum(np.linalg.norm(r1, axis=-1), 1e-12)
    den = np.sum(cr**2, -1)+core**2
    k = (1+(r1@d)/n1)/den/(4*np.pi)
    return cr*k[..., None]


def horseshoe(P, A, B, d, core):
    """Vitesse induite en P par chaque fer à cheval (Gamma = 1) : sillage le long de d."""
    return _semi_infini(P, B, d, core)-_semi_infini(P, A, d, core)+_segment_fini(P, A, B, core)


def induced_drag_trefftz(A, B, G, d, nx, ny, rho):
    """Traînée induite par le bilan d'énergie dans le plan de Trefftz (très loin derrière l'aile).
    Les filaments de sillage y sont des tourbillons 2D ; D = -(rho/2) * somme(Gamma_s * w_n * dl)
    avec w_n la vitesse normale induite au milieu de chaque bande par tous les filaments."""
    Ag, Bg = A.reshape(nx, ny, 3), B.reshape(nx, ny, 3)
    q = np.empty((ny+1, 3))                                   # position des filaments de sillage
    q[0], q[ny] = Ag[:, 0].mean(0), Bg[:, ny-1].mean(0)
    q[1:ny] = 0.5*(Bg[:, :ny-1].mean(0)+Ag[:, 1:].mean(0))
    q = q-np.outer(q@d, d)                                    # projection dans le plan perpendiculaire à d
    Gs = G.reshape(nx, ny).sum(0)                             # circulation de chaque bande d'envergure
    Gp = np.r_[0.0, Gs, 0.0]
    kappa = Gp[:-1]-Gp[1:]                                    # circulation nette de chaque filament
    t = q[1:]-q[:-1]
    dl = np.linalg.norm(t, axis=1)
    nrm = np.cross(d, t/dl[:, None])                          # normale à chaque bande, dans le plan
    rr = 0.5*(q[1:]+q[:-1])[:, None, :]-q[None, :, :]         # milieu de bande -> filament
    v = np.cross(d, rr)/np.sum(rr**2, -1)[..., None]/(2*np.pi)
    w_n = np.sum(np.einsum("k,skj->sj", kappa, v)*nrm, -1)
    return float(-(rho/2)*np.sum(Gs*w_n*dl))


def solve_vlm(geom, nx, ny, V, d, rho):
    A, B, C, n, S, P = build_panels(geom, nx, ny)
    U = V*d
    M = np.einsum("ijk,ik->ij", horseshoe(C, A, B, d, CORE_EXACT), n)   # A_ij = v_j(x_i).n_i
    rhs = -n@U                                                           # u.n = 0 sur l'aile
    G = np.linalg.solve(M, rhs)
    # Kutta-Joukowski sur chaque segment porteur, avec la vitesse locale au milieu du segment
    mid = 0.5*(A+B)
    u_mid = U+np.einsum("ijk,j->ik", horseshoe(mid, A, B, d, CORE_EXACT), G)
    F = rho*G[:, None]*np.cross(u_mid, B-A)
    lift_dir = np.cross(d, [0.0, 1.0, 0.0])
    lift_dir /= np.linalg.norm(lift_dir)
    L = float(F.sum(0)@lift_dir)                                         # portance : Kutta-Joukowski
    D = induced_drag_trefftz(A, B, G, d, nx, ny, rho)                    # traînée induite : Trefftz
    q = 0.5*rho*V**2
    G_strip = G.reshape(nx, ny).sum(0)
    return dict(A=A, B=B, C=C, n=n, G=G, U=U, d=d, V=V, S=float(S), AR=float(geom[0]**2/S),
                L=L, D=D, CL=L/(q*S), CDi=D/(q*S), P=P, G_strip=G_strip,
                y_strip=0.5*(P[0, :-1, 1]+P[0, 1:, 1]))


def velocity(pts, r, core, chunk=400):
    """Champ de vitesse total u(x) = U_inf + somme des fers à cheval (Biot-Savart)."""
    out = np.empty((len(pts), 3))
    for k in range(0, len(pts), chunk):
        v = horseshoe(pts[k:k+chunk], r["A"], r["B"], r["d"], core)
        out[k:k+chunk] = r["U"]+np.einsum("mnk,n->mk", v, r["G"])
    return out


def integrate_rk4(r, seeds, n_steps, dt, core):
    """dX/dt = u(X) intégré par Runge-Kutta d'ordre 4. Renvoie positions et |u|/V."""
    X = seeds.copy()
    traj = np.empty((n_steps+1, len(seeds), 3))
    speed = np.empty((n_steps+1, len(seeds)))
    f = lambda Y: velocity(Y, r, core)
    traj[0] = X
    for k in range(n_steps):
        k1 = f(X)
        speed[k] = np.linalg.norm(k1, axis=1)/r["V"]
        k2 = f(X+0.5*dt*k1)
        k3 = f(X+0.5*dt*k2)
        k4 = f(X+dt*k3)
        X = X+dt/6*(k1+2*k2+2*k3+k4)
        traj[k+1] = X
    speed[n_steps] = np.linalg.norm(f(X), axis=1)/r["V"]
    return traj, speed


# ---------- Fonctions mises en cache -----------------------------------
@st.cache_data(show_spinner="Résolution du système VLM…")
def compute_vlm(geom, nx, ny, flow):
    airspeed, wind_speed, wind_dir, aoa, rho = flow
    V, d, beta = freestream(airspeed, wind_speed, wind_dir, aoa)
    r = solve_vlm(geom, nx, ny, V, d, rho)
    r["beta"] = beta
    return r


@st.cache_data(show_spinner="Intégration RK4 des trajectoires…")
def compute_pathlines(geom, nx, ny, flow, n_lines, n_steps=200):
    r = compute_vlm(geom, nx, ny, flow)
    c_root, c_tip, sweep, dihedral = geom[1], geom[2], geom[3], geom[4]
    c_mean, semi, d = 0.5*(c_root+c_tip), geom[0]/2, r["d"]
    lift = np.cross(d, [0.0, 1.0, 0.0])
    lift /= np.linalg.norm(lift)
    # Points de départ en amont du bord d'attaque, sur plusieurs niveaux verticaux
    ys = np.linspace(-0.95*semi, 0.95*semi, n_lines)
    base = np.column_stack([np.abs(ys)*np.tan(np.deg2rad(sweep)), ys,
                            np.abs(ys)*np.tan(np.deg2rad(dihedral))])
    L_up = 0.6*c_root
    seeds = np.vstack([base-L_up*d+off*c_mean*lift for off in (-0.30, -0.12, 0.12, 0.30)])
    x_ext = r["P"][..., 0].max()-r["P"][..., 0].min()
    dt = (L_up+x_ext+1.5*c_root)/n_steps/r["V"]
    traj, speed = integrate_rk4(r, seeds, n_steps, dt, core=0.004*c_mean)
    return dict(traj=traj, speed=speed, dt=dt)


@st.cache_data(show_spinner="Calcul du champ de vitesse (plan de coupe)…")
def compute_slice(geom, nx, ny, flow, y_cut):
    r = compute_vlm(geom, nx, ny, flow)
    s = np.linspace(0, 1, 60)
    sec = section_points(abs(y_cut), s, geom)
    chord = geom[1]+(geom[2]-geom[1])*abs(y_cut)/(geom[0]/2)
    z0 = abs(y_cut)*np.tan(np.deg2rad(geom[4]))
    xs = np.linspace(sec["camber"][0].min()-0.7*chord, sec["camber"][0].max()+1.2*chord, 110)
    zs = np.linspace(z0-0.9*chord, z0+0.9*chord, 70)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.column_stack([XX.ravel(), np.full(XX.size, y_cut), ZZ.ravel()])
    u = velocity(pts, r, core=0.004*0.5*(geom[1]+geom[2]))
    return dict(xs=xs, zs=zs, q=(np.linalg.norm(u, axis=1)/r["V"]).reshape(XX.shape),
                upper=sec["upper"], lower=sec["lower"])


def verifications(r, geom):
    """Contrôles numériques : u.n aux points de contrôle, div u et rot u par différences finies."""
    c_mean, V = 0.5*(geom[1]+geom[2]), r["V"]
    un = np.einsum("ik,ik->i", velocity(r["C"], r, CORE_EXACT), r["n"])
    nodes = r["P"].reshape(-1, 3)
    lo, hi = nodes.min(0), nodes.max(0)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(lo[0]-0.3*c_mean, hi[0]+c_mean, 8),
                           rng.uniform(lo[1], hi[1], 8),
                           0.5*(lo[2]+hi[2])+rng.choice([-1, 1], 8)*rng.uniform(0.3, 0.8, 8)*c_mean])
    h, J = 1e-4, np.zeros((8, 3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[:, :, j] = (velocity(pts+e, r, CORE_EXACT)-velocity(pts-e, r, CORE_EXACT))/(2*h)
    div = J[:, 0, 0]+J[:, 1, 1]+J[:, 2, 2]
    curl = np.stack([J[:, 2, 1]-J[:, 1, 2], J[:, 0, 2]-J[:, 2, 0], J[:, 1, 0]-J[:, 0, 1]], 1)
    scale = V/c_mean
    return (np.abs(un).max()/V, np.abs(div).max()/scale, np.linalg.norm(curl, axis=1).max()/scale)


# ======================================================================
# 3. FIGURES DE L'ÉCOULEMENT
# ======================================================================
def flow_figure(surfaces, r, pl, wind_speed, wind_dir, c_root):
    fig = make_figure(surfaces, False, "Écoulement relatif autour de l'aile — lignes de courant "
                                       "et vecteurs de vitesse (RK4)")
    traj, speed = pl["traj"], pl["speed"]
    n_steps, n_seeds = traj.shape[0]-1, traj.shape[1]

    def arrow_segments(positions, vectors):
        """Construit de petites flèches 3D en segments, sans cônes volumineux."""
        tails, heads = positions, positions+vectors
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors/np.maximum(lengths[:, None], 1e-12)
        reference = np.zeros_like(directions)
        reference[:, 2] = 1.0
        parallel = np.abs(directions[:, 2]) > 0.9
        reference[parallel] = [0.0, 1.0, 0.0]
        side = np.cross(directions, reference)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
        head_length = np.minimum(0.025*c_root, 0.35*lengths)
        wing = 0.65*head_length[:, None]*side
        back = heads-directions*head_length[:, None]
        x, y, z = [], [], []
        for tail, head, back_point, wing_point in zip(tails, heads, back, wing):
            left = back_point+wing_point
            right = back_point-wing_point
            x.extend([tail[0], head[0], None, head[0], left[0], None, head[0], right[0], None])
            y.extend([tail[1], head[1], None, head[1], left[1], None, head[1], right[1], None])
            z.extend([tail[2], head[2], None, head[2], left[2], None, head[2], right[2], None])
        return x, y, z

    lx, ly, lz = [], [], []                                   # lignes de courant
    for m in range(n_seeds):
        lx += traj[:, m, 0].tolist()+[None]
        ly += traj[:, m, 1].tolist()+[None]
        lz += traj[:, m, 2].tolist()+[None]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name="Lignes de courant",
                               line=dict(color="rgba(30,90,160,0.55)", width=2), hoverinfo="skip"))

    # Flèches vectorielles orientées selon la vitesse locale de l'écoulement.
    arrow_count = min(6, max(2, n_steps // 25))
    line_ids = np.linspace(0, n_seeds - 1, min(5, n_seeds), dtype=int)
    step_ids = np.linspace(0, n_steps - 1, arrow_count, dtype=int)
    arrow_positions, arrow_vectors = [], []
    for line_id in line_ids:
        for step_id in step_ids:
            next_id = (step_id + 1) % n_steps
            vector = traj[next_id, line_id] - traj[step_id, line_id]
            norm = np.linalg.norm(vector)
            if norm > 1e-12:
                arrow_positions.append(traj[step_id, line_id])
                arrow_vectors.append(vector / norm * 0.12 * c_root)
    if arrow_positions:
        arrow_positions = np.asarray(arrow_positions)
        arrow_vectors = np.asarray(arrow_vectors)
        ax, ay, az = arrow_segments(arrow_positions, arrow_vectors)
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az, mode="lines", name="Vecteurs de vitesse",
            line=dict(color="#00ACC1", width=4), hoverinfo="skip"
        ))

    if wind_speed > 0:                                        # flèche du vent (repère sol)
        z_top = max(np.nanmax(Z) for _, _, Z in surfaces.values())
        u_w = np.array([np.cos(np.deg2rad(wind_dir)), np.sin(np.deg2rad(wind_dir)), 0.0])
        mid = np.array([0.5*c_root, 0.0, z_top+0.25*c_root+0.05])
        tail, head = mid-0.4*u_w, mid+0.4*u_w
        fig.add_trace(go.Scatter3d(
            x=[tail[0], head[0]], y=[tail[1], head[1]], z=[tail[2], head[2]],
            mode="lines+markers", name="Vent (repère sol)",
            line=dict(color="#E67E22", width=7), marker=dict(size=[0, 6], color="#E67E22")))

    # Lâcher continu : les anciens points-particules sont remplacés par des flèches animées.
    per_line = 5
    spacing = max(1, n_steps//per_line)
    n_frames = 3*spacing

    def animated_vectors(frame):
        indices = (np.arange(per_line)*spacing+frame) % n_steps
        line_ids = np.arange(n_seeds)[None, :]
        positions = traj[indices[:, None], line_ids].reshape(-1, 3)
        next_indices = (indices+1) % n_steps
        next_positions = traj[next_indices[:, None], line_ids].reshape(-1, 3)
        vectors = next_positions-positions
        norms = np.linalg.norm(vectors, axis=1)
        vectors = np.divide(vectors, np.maximum(norms[:, None], 1e-12))*0.08*c_root
        return positions, vectors

    p0, v0 = animated_vectors(0)
    animated_trace = len(fig.data)
    ax, ay, az = arrow_segments(p0, v0)
    fig.add_trace(go.Scatter3d(
        x=ax, y=ay, z=az, mode="lines", name="Vecteurs animés",
        line=dict(color="#FDD835", width=4), hoverinfo="skip"
    ))
    frames = []
    for frame in range(n_frames):
        positions, vectors = animated_vectors(frame)
        ax, ay, az = arrow_segments(positions, vectors)
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=ax, y=ay, z=az,
                mode="lines", line=dict(color="#FDD835", width=4),
                hoverinfo="skip"
            )],
            traces=[animated_trace], name=str(frame)))
    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="▶ Lecture", method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         transition=dict(duration=0),
                                         mode="immediate")])])],
        legend=dict(x=0.02, y=0.98))
    return fig


def slice_figure(sl, quantity, y_cut):
    q = sl["q"]
    if quantity == "Pression (Cp, Bernoulli)":
        z, scale, lo, hi, title = 1-q**2, "RdBu", -1.0, 1.0, "Cp = 1 − |u|²/V∞²"
    else:
        z, scale, lo, hi, title = q, "Viridis", 0.7, 1.3, "|u| / V∞"
    fig = go.Figure(go.Contour(
        x=sl["xs"], y=sl["zs"], z=np.clip(z, lo, hi), colorscale=scale, zmin=lo, zmax=hi,
        contours=dict(start=lo, end=hi, size=(hi-lo)/30, coloring="heatmap"),
        line=dict(width=0), colorbar=dict(title=title)))
    xu, zu = sl["upper"]
    xl, zl = sl["lower"]
    fig.add_trace(go.Scatter(x=np.r_[xu, xl[::-1]], y=np.r_[zu, zl[::-1]], mode="lines",
                             fill="toself", fillcolor="rgba(60,60,60,0.85)",
                             line=dict(color="black", width=1), hoverinfo="skip", showlegend=False))
    fig.update_layout(title=f"Plan de coupe y = {y_cut:.2f} m", height=430,
                      margin=dict(l=0, r=0, t=45, b=0),
                      xaxis_title="x (m)", yaxis_title="z (m)", yaxis=dict(scaleanchor="x"))
    return fig


def span_loading_figure(r):
    y, g = r["y_strip"], r["G_strip"]
    edges = r["P"][0, :, 1]                                   # bords des bandes
    b = edges[-1]-edges[0]
    g0 = 4*np.sum(g*np.diff(edges))/(np.pi*b)                 # même circulation totale qu'une loi elliptique
    y_ell = np.linspace(-b/2, b/2, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=g, mode="lines+markers", name="Γ(y) — VLM"))
    fig.add_trace(go.Scatter(x=y_ell, y=g0*np.sqrt(np.maximum(1-(2*y_ell/b)**2, 0)),
                             mode="lines", line=dict(dash="dash"), name="Loi elliptique (même portance)"))
    fig.update_layout(title="Circulation liée en envergure", height=380,
                      margin=dict(l=0, r=0, t=45, b=0),
                      xaxis_title="y (m)", yaxis_title="Γ (m²/s)")
    return fig


def span_resolution_figure(span, n_span, ny_vlm):
    """Représentation 1D des points de discrétisation sur toute l'envergure."""
    semi = span/2
    surface_nodes = np.linspace(-semi, semi, 2*n_span+1)
    vlm_edges = -(semi)*np.cos(np.pi*np.arange(ny_vlm+1)/ny_vlm)
    vlm_centers = 0.5*(vlm_edges[:-1]+vlm_edges[1:])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=surface_nodes, y=np.zeros_like(surface_nodes), mode="markers",
        name=f"Maillage surfacique ({2*n_span} divisions)",
        marker=dict(size=8, color="#1565C0")))
    fig.add_trace(go.Scatter(
        x=vlm_edges, y=np.ones_like(vlm_edges), mode="markers",
        name=f"Bords des panneaux VLM ({ny_vlm} panneaux)",
        marker=dict(size=8, color="#E67E22", symbol="line-ns")))
    fig.add_trace(go.Scatter(
        x=vlm_centers, y=np.ones_like(vlm_centers), mode="markers",
        name="Points centraux des panneaux VLM",
        marker=dict(size=6, color="#8E44AD", symbol="diamond")))
    fig.update_layout(
        title="Discrétisation 1D de l'envergure",
        height=360,
        margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="Position en envergure y (m)",
        yaxis=dict(
            tickvals=[0, 1], ticktext=["Maillage surface", "Panneaux VLM"],
            range=[-0.35, 1.35], fixedrange=True),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2))
    return fig


def fortran_profile_figure(profile):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile["y_m"], y=profile["gamma_sur_V_m"], mode="lines",
        name="Γ / V∞", line=dict(color="#5A2A05", width=3)))
    fig.update_layout(
        title="Profil de circulation calculé par Fortran",
        height=380, margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="y (m)", yaxis_title="Γ / V∞ (m)",
        legend=dict(x=0.02, y=0.98))
    return fig


def fortran_gradient_figure(gradient):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gradient["y_noeud_m"],
        y=gradient["gradient_gamma_sur_V_par_m"],
        mode="lines+markers",
        name="d(Γ/V∞)/dy",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=5, color=gradient["rang_proprietaire"],
                    colorscale="Turbo", colorbar=dict(title="Rang MPI"))))
    fig.update_layout(
        title="Gradient de circulation échangé entre les processus MPI",
        height=400, margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="Position du nœud y (m)",
        yaxis_title="Gradient d(Γ/V∞)/dy (1/s)",
        legend=dict(x=0.02, y=0.98))
    return fig


def wing_resolution_figure(geom, n_span, ny_vlm):
    """Schéma plan annoté des variables géométriques et des subdivisions."""
    span, root_chord, tip_chord, sweep, _, _, _ = geom
    semi = span/2
    x_tip = semi*np.tan(np.deg2rad(sweep))
    x_root_te = root_chord
    x_tip_te = x_tip+tip_chord
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, x_root_te, x_tip_te, x_tip, 0, x_tip, x_tip_te, x_root_te, 0],
        y=[0, 0, semi, semi, 0, -semi, -semi, 0, 0],
        fill="toself", mode="lines", name="Aile",
        fillcolor="rgba(90,42,5,0.16)", line=dict(color="#5A2A05", width=3)))

    panel_y = np.linspace(-semi, semi, min(2*n_span+1, 41))
    for y in panel_y[1:-1]:
        frac = abs(y)/semi
        x_le = frac*x_tip
        chord = root_chord+(tip_chord-root_chord)*frac
        fig.add_trace(go.Scatter(
            x=[x_le, x_le+chord], y=[y, y], mode="lines",
            line=dict(color="#D98B2B", width=1), showlegend=False, hoverinfo="skip"))

    def arrow(x, y, ax, ay, color, label, label_x, label_y):
        fig.add_annotation(x=ax, y=ay, ax=x, ay=y, xref="x", yref="y",
                           axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor=color)
        fig.add_annotation(x=label_x, y=label_y, text=label, showarrow=False,
                           font=dict(color=color, size=13))

    arrow(-0.02, 0, -0.02, semi, "#1565C0", "b / 2", -0.14, semi/2)
    arrow(0, -semi-0.08, 0, semi+0.08, "#1565C0", "b = envergure", -0.25, 0)
    arrow(0, -0.12, root_chord, -0.12, "#C62828", "c_r", root_chord/2, -0.2)
    arrow(x_tip, semi+0.1, x_tip_te, semi+0.1, "#C62828", "c_t", x_tip+tip_chord/2, semi+0.18)
    arrow(0, semi+0.28, x_tip, semi+0.28, "#6A1B9A", "flèche Λ", x_tip/2, semi+0.38)
    arrow(0.35*root_chord, -semi*0.68, 0.35*root_chord, -semi*0.68,
          "#D98B2B", f"{ny_vlm} panneaux VLM", 0.35*root_chord, -semi*0.78)
    fig.add_annotation(x=root_chord*0.55, y=semi*0.45, text="x : corde", showarrow=False,
                       font=dict(color="#263238", size=12))
    fig.add_annotation(x=x_tip*0.55+tip_chord*0.25, y=semi*0.72, text="y : envergure",
                       showarrow=False, font=dict(color="#263238", size=12))
    fig.update_layout(
        title="Aile en plan : variables utilisées par la discrétisation",
        height=560, margin=dict(l=20, r=20, t=55, b=35),
        xaxis_title="x (m) — direction de la corde",
        yaxis_title="y (m) — direction de l'envergure",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(zeroline=True, zerolinecolor="#90A4AE"),
        legend=dict(orientation="h", y=-0.12))
    return fig


# ======================================================================
# 4. INTERFACE
# ======================================================================
with st.sidebar:
    st.header("Paramètres géométriques")
    span = st.number_input("Envergure totale (m)", 0.2, 30.0, 2.0, 0.1)
    root_chord = st.number_input("Corde à la racine (m)", 0.05, 10.0, 0.35, 0.01)
    tip_chord = st.number_input("Corde au saumon (m)", 0.03, 10.0, 0.18, 0.01)
    sweep = st.slider("Flèche du bord d'attaque (°)", -10, 45, 12)
    dihedral = st.slider("Dièdre (°)", -5, 20, 4)
    twist = st.slider("Torsion au saumon (°)", -15, 15, -2)
    naca = st.text_input("Profil NACA à 4 chiffres", "0012")

    st.header("Paramètres du maillage")
    n_span = st.slider("Divisions en envergure (demi-aile)", 4, 100, 24)
    n_chord = st.slider("Divisions en corde", 4, 100, 30)
    element_type = st.selectbox("Type d'éléments", ["Triangles", "Quadrilatères"])

    st.header("Conditions de vol")
    airspeed = st.slider("Vitesse de l'aile (m/s)", 1.0, 30.0, 10.0, 0.5)
    wind_speed = st.slider("Vitesse du vent (m/s)", 0.0, 30.0, 5.0, 0.5)
    wind_direction = st.slider("Direction du vent (°) : 0 = de face, ±90 = de côté, 180 = arrière",
                               -180, 180, 0)
    angle_of_attack = st.slider("Angle d'attaque (°)", -10, 20, 5)
    rho = st.number_input("Masse volumique de l'air (kg/m³)", 0.5, 1.5, 1.225, 0.005, format="%.3f")

    st.header("Paramètres VLM")
    nx_vlm = st.slider("Panneaux en corde", 1, 10, 4)
    ny_vlm = st.slider("Panneaux en envergure (pair)", 8, 60, 24, step=2)
    n_lines = st.slider("Lignes de courant par niveau", 5, 25, 13)

geom = (float(span), float(root_chord), float(tip_chord), float(sweep), float(dihedral),
        float(twist), str(naca))
flow = (float(airspeed), float(wind_speed), float(wind_direction), float(angle_of_attack), float(rho))

try:
    surfaces = build_wing_surface(geom, n_span, n_chord)
    nodes, elements = mesh_from_surfaces(surfaces, element_type)
except Exception as exc:
    st.error(f"Paramètres invalides : {exc}")
    st.stop()

tab_geo, tab_flow, tab_solveur = st.tabs([
    "Géométrie et maillage",
    "Écoulement (VLM)",
    "Solveur",
])

# ---------------------------------------------------------------- Géométrie
with tab_geo:
    st.subheader("Géométrie et maillage")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_figure(surfaces, False, "Aile initiale"), **STRETCH)
    with col2:
        st.plotly_chart(make_figure(surfaces, True, "Aile maillée"), **STRETCH)

    k1, k2, k3 = st.columns(3)
    k1.metric("Nœuds", f"{len(nodes):,}".replace(",", " "))
    k2.metric("Éléments", f"{len(elements):,}".replace(",", " "))
    k3.metric("Type", element_type)

    st.subheader("Export du maillage")
    node_df = pd.DataFrame(nodes, columns=["x_m", "y_m", "z_m"])
    node_df.index.name = "node_id"
    elem_df = pd.DataFrame(elements, columns=[f"node_{i+1}" for i in range(elements.shape[1])])
    elem_df.index.name = "element_id"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Télécharger les nœuds (CSV)", node_df.to_csv().encode("utf-8"),
                           "aile_noeuds.csv", "text/csv")
    with c2:
        st.download_button("Télécharger les éléments (CSV)", elem_df.to_csv().encode("utf-8"),
                           "aile_elements.csv", "text/csv")
    with st.expander("Aperçu des nœuds"):
        st.dataframe(node_df.head(20), **STRETCH)
    with st.expander("Aperçu des éléments"):
        st.dataframe(elem_df.head(20), **STRETCH)
    st.info("Maillage surfacique structuré de l'extrados et de l'intrados (pas de maillage volumique).")

# ------------------------------------------------- Calcul VLM (une seule fois)
vlm, vlm_error = None, None
try:
    vlm = compute_vlm(geom, nx_vlm, ny_vlm, flow)
except ValueError as exc:
    vlm_error = str(exc)

# ------------------------------------------------------------------ Écoulement
with tab_flow:
    st.subheader("Écoulement potentiel autour de l'aile (VLM)")
    if vlm is None:
        st.error(vlm_error)
    else:
        if abs(vlm["beta"]) > 15 or abs(angle_of_attack) > 12:
            st.warning("Dérapage > 15° ou incidence > 12° : le modèle linéaire (sillage plan, "
                       "pas de décrochage) devient peu fiable.")
        st.caption(f"Écoulement relatif : V∞ = {vlm['V']:.1f} m/s, incidence α = {angle_of_attack:.1f}°, "
                   f"dérapage β = {vlm['beta']:.1f}°. Repère de l'aile : l'aile avance vers −x.")

        pl = compute_pathlines(geom, nx_vlm, ny_vlm, flow, n_lines)
        st.plotly_chart(flow_figure(surfaces, vlm, pl, wind_speed, wind_direction, root_chord), **STRETCH)
        st.caption(f"Les flèches indiquent la direction locale de la vitesse. Intégration RK4 avec un pas "
                   f"de {pl['dt']*1000:.2f} ms de temps physique. "
                   "Le flux est stationnaire dans le repère de l'aile : trajectoires et lignes de courant "
                   "coïncident. L'aile affichée est épaisse, mais le calcul porte sur sa surface moyenne.")

        st.subheader("Champ de vitesse par Biot–Savart : plan de coupe")
        semi = geom[0]/2
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            y_cut = st.slider("Position du plan de coupe y (m)", -float(semi), float(semi),
                              float(round(0.3*semi, 2)), 0.01)
        with cc2:
            quantity = st.radio("Grandeur affichée", ["Vitesse |u|/V∞", "Pression (Cp, Bernoulli)"],
                                horizontal=True)
        sl = compute_slice(geom, nx_vlm, ny_vlm, flow, float(y_cut))
        st.plotly_chart(slice_figure(sl, quantity, y_cut), **STRETCH)
        st.caption("Les disques autour de l'aile sont la signature des filaments tourbillonnaires "
                   "discrets (vitesse en 1/r) : c'est une propriété du modèle discret, pas un phénomène "
                   "physique. Le saut de vitesse à travers l'aile et son sillage est celui de la nappe "
                   "tourbillonnaire. Couleurs saturées en dehors de l'échelle.")

# --------------------------------------------------------------- Résolution
with tab_solveur:
    st.subheader("Solveur — résolution, résultats et vérifications")
    st.info(
        f"Sections de l'aile : {2*n_span} en envergure × {n_chord} en corde | "
        f"Panneaux VLM : {nx_vlm} × {ny_vlm} = {nx_vlm*ny_vlm} | "
        "Les résultats aérodynamiques de cet onglet proviennent du solveur Fortran."
    )
    st.markdown(
        "Cette vue 1D montre directement l'effet des paramètres de discrétisation : "
        "`Divisions en envergure (demi-aile)` contrôle le maillage surfacique, tandis que "
        "`Panneaux en envergure (pair)` contrôle les panneaux utilisés par le calcul VLM."
    )
    st.plotly_chart(wing_resolution_figure(geom, n_span, ny_vlm), **STRETCH)
    st.markdown("""
    **Lecture des couleurs du schéma**

    <span style="color:#1565C0"><b>Bleu :</b> b, b/2 et direction y (envergure).</span>
    <span style="color:#C62828"><b>Rouge :</b> c<sub>r</sub> et c<sub>t</sub> (cordes).</span>
    <span style="color:#6A1B9A"><b>Violet :</b> Λ, flèche du bord d'attaque.</span>
    <span style="color:#D98B2B"><b>Orange :</b> lignes de découpage et panneaux VLM.</span>
    """, unsafe_allow_html=True)
    st.markdown("### Équations de la discrétisation")
    eq1, eq2 = st.columns(2)
    with eq1:
        st.latex(r"c(y)=c_r+\frac{|y|}{b/2}\left(c_t-c_r\right)")
        st.caption("La corde locale c(y) varie entre la corde à la racine cᵣ et celle au saumon cₜ.")
    with eq2:
        st.latex(r"\Delta y_{\mathrm{moy}}=\frac{b}{2\,n_{\mathrm{span}}},\qquad "
                 r"N_{\mathrm{VLM}}=n_x\,n_y")
        st.caption("n_span définit le maillage de surface ; n_y et n_x définissent les panneaux du calcul VLM.")
    st.markdown("### Équation intégrale de portance de Prandtl")
    st.markdown(
        "La résolution en envergure consiste à déterminer la circulation "
        r"$\Gamma(x)$ sur toute l'aile. L'équation de Prandtl relie l'incidence "
        "locale à la circulation et à l'influence du sillage :"
    )
    st.latex(
        r"\alpha(x)=\frac{\Gamma(x)}{2\,b\,V_\infty}"
        r"+\frac{1}{4\pi V_\infty}\,\mathrm{PV}\!\int_{-b}^{b}"
        r"\frac{\mathrm{d}\Gamma(\tau)}{\mathrm{d}\tau}\,"
        r"\frac{1}{x-\tau}\,\mathrm{d}\tau"
    )
    st.markdown("""
    <div style="border-left:4px solid #5A2A05;padding:.55rem .8rem;
                background:rgba(90,42,5,.06);line-height:1.75;">
    <b style="color:#C62828">α(x)</b> : incidence locale au point d'envergure
    <b style="color:#1565C0">x</b>.<br>
    <b style="color:#D98B2B">Γ(x)</b> : circulation locale autour du profil,
    directement liée à la portance par Kutta–Joukowski.<br>
    <b style="color:#6A1B9A">b</b> : envergure totale de l'aile et
    <b style="color:#00838F">V<sub>∞</sub></b> : vitesse de l'écoulement incident.<br>
    <b style="color:#00838F">τ</b> : position d'un élément de l'envergure qui
    influence le point <b style="color:#1565C0">x</b>.<br>
    <b style="color:#E65100">dΓ(τ)/dτ</b> : variation de circulation ; elle crée
    le sillage tourbillonnaire. <b>PV</b> indique la valeur principale de Cauchy
    au point singulier <b>x = τ</b>.
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Dans le modèle discret, l'intégrale est remplacée par une somme sur les "
        f"{ny_vlm} panneaux d'envergure VLM ; augmenter cette résolution décrit plus "
        "finement la variation de Γ près des saumons."
    )
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Divisions surface", f"{2*n_span} sur l'aile entière")
    rc2.metric("Panneaux VLM", f"{ny_vlm}")
    rc3.metric("Pas moyen surface", f"{span/(2*n_span):.4f} m")
    st.plotly_chart(span_resolution_figure(span, n_span, ny_vlm), **STRETCH)
    st.info(
        "Les points orange sont espacés selon la loi cosinus du VLM : ils sont plus concentrés "
        "près des saumons pour mieux représenter les effets marginaux."
    )

    st.divider()
    st.subheader("Calcul scientifique Fortran/MPI")
    st.caption(
        "Cette section pilote le solveur Fortran de Prandtl réellement présent dans "
        "`projet_prandtl_mpi`. Les résultats et journaux ci-dessous sont propres à ce calcul."
    )
    tools = tool_status()
    status_cols = st.columns(2)
    status_cols[0].write(
        f"Compilateur MPI Fortran : "
        f"`{tools['compilateur MPI Fortran'] or 'introuvable'}`"
    )
    status_cols[1].write(
        f"Lanceur MPI : `{tools['lanceur MPI'] or 'introuvable'}`"
    )

    with st.form("fortran_simulation_form"):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            fortran_n = st.number_input(
                "Sections Fortran N", min_value=2, max_value=20000,
                value=max(2, 2*int(n_span)),
                step=2, help="N est le nombre de sections/panneaux de la ligne portante ; il est pair.")
        with f2:
            fortran_processes = st.number_input(
                "Processus MPI (exécution)", min_value=1, max_value=128, value=1, step=1,
                help="Nombre de processus utilisés par mpiexec pour lancer le solveur compilé.")
        with f3:
            fortran_repetitions = st.number_input(
                "Répétitions chronométrées", min_value=1, max_value=20, value=3, step=1)
        with f4:
            fortran_overlap = st.checkbox(
                "Recouvrement MPI", value=True,
                help="Iallgatherv non bloquant + calcul local recouvert.")
        fortran_dump = st.checkbox(
            "Récupérer le profil Γ(y) produit par Fortran", value=True)
        f5, f6, f7, f8 = st.columns(4)
        with f5:
            fortran_alpha_l0 = st.number_input(
                "αL=0 (°)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
        with f6:
            fortran_a0 = st.number_input(
                "Pente a₀ (1/rad)", min_value=0.1, max_value=20.0,
                value=float(2*np.pi), step=0.01)
        with f7:
            fortran_tol = st.number_input(
                "Tolérance", min_value=1e-14, max_value=1e-2,
                value=1e-10, format="%.1e")
        with f8:
            fortran_itmax = st.number_input(
                "Itérations max", min_value=1, max_value=100000, value=5000, step=100)
        fortran_elliptique = st.checkbox(
            "Corde elliptique Fortran", value=False,
            help="Désactivé : Fortran utilise la corde racine/saumon de l'interface.")
        run_fortran = st.form_submit_button(
            "Compiler et lancer le solveur Fortran", type="primary")
    fortran_vinf = float(np.hypot(
        airspeed + wind_speed*np.cos(np.deg2rad(wind_direction)),
        wind_speed*np.sin(np.deg2rad(wind_direction))))
    fortran_n_effective = int(fortran_n) + int(fortran_n) % 2
    st.caption(
        f"Configuration sélectionnée : {fortran_n_effective} sections Fortran, "
        f"{int(fortran_processes)} processus MPI, {int(fortran_repetitions)} répétitions, "
        f"V∞ Fortran = {fortran_vinf:.3f} m/s."
    )
    st.warning(
        "Le solveur Fortran de Prandtl utilise b, cᵣ, cₜ, α, vrillage, αL=0, a₀, "
        "V∞ et ρ. La flèche, le dièdre, le profil NACA et la direction latérale du vent "
        "restent propres au VLM 3D et ne sont pas utilisés par le modèle Fortran."
    )
    with st.expander("Correspondance exacte des paramètres transmis au Fortran"):
        parameter_map = pd.DataFrame([
            ["Envergure totale", "envergure", f"{span:.6g} m", "transmis"],
            ["Corde racine", "croot", f"{root_chord:.6g} m", "transmis"],
            ["Corde saumon", "ctip", f"{tip_chord:.6g} m", "transmis"],
            ["Angle d'attaque", "incidence", f"{angle_of_attack:.6g}°", "transmis"],
            ["Torsion au saumon", "vrillage", f"{twist:.6g}°", "transmis"],
            ["Incidence portance nulle", "alpha_l0", f"{fortran_alpha_l0:.6g}°", "transmis"],
            ["Pente 2D", "a0", f"{fortran_a0:.6g} 1/rad", "transmis"],
            ["Vitesse résultante", "vinf", f"{fortran_vinf:.6g} m/s", "transmis"],
            ["Masse volumique", "rho", f"{rho:.6g} kg/m³", "transmis"],
            ["Sections", "n", f"{fortran_n_effective}", "transmis, pair"],
            ["Tolérance", "tol", f"{fortran_tol:.2e}", "transmis"],
            ["Itérations max", "itmax", f"{int(fortran_itmax)}", "transmis"],
            ["Flèche / dièdre / NACA", "—", "—", "non représentés par le modèle 1D"],
        ], columns=["Entrée Streamlit", "Clé Fortran", "Valeur", "Statut"])
        st.dataframe(parameter_map, hide_index=True, **STRETCH)

    if run_fortran:
        fortran_parameters = {
            "n": fortran_n_effective,
            "envergure": float(span),
            "croot": float(root_chord),
            "ctip": float(tip_chord),
            "incidence": float(angle_of_attack),
            "vrillage": float(twist),
            "alpha_l0": float(fortran_alpha_l0),
            "a0": float(fortran_a0),
            "elliptique": int(fortran_elliptique),
            "vinf": fortran_vinf,
            "rho": float(rho),
            "tol": float(fortran_tol),
            "itmax": int(fortran_itmax),
            "rep": int(fortran_repetitions),
            "recouvrement": int(fortran_overlap),
        }
        try:
            with st.spinner("Compilation du solveur Fortran MPI…"):
                compilation = compile_solver()
            with st.spinner("Exécution du solveur Fortran MPI…"):
                execution = run_solver(
                    fortran_parameters, int(fortran_processes), bool(fortran_dump))
            st.session_state["fortran_result"] = {
                "compilation": compilation, "execution": execution}
            st.session_state.pop("fortran_error", None)
            st.success("Compilation et exécution Fortran terminées.")
        except FortranBridgeError as exc:
            st.session_state["fortran_error"] = str(exc)
            st.session_state.pop("fortran_result", None)

    if "fortran_error" in st.session_state:
        st.error("Calcul Fortran impossible")
        st.code(st.session_state["fortran_error"], language="text")

    if "fortran_result" in st.session_state:
        fortran_result = st.session_state["fortran_result"]
        compilation = fortran_result["compilation"]
        execution = fortran_result["execution"]
        metrics = execution["metrics"]
        st.subheader("Résultats Fortran")
        result_row = metrics.iloc[0]
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Cₗ Fortran", f"{result_row['CL']:.6f}")
        m2.metric("C_Di Fortran", f"{result_row['CDi']:.6f}")
        m3.metric("Résidu", f"{result_row['residu']:.2e}")
        m4.metric("Itérations", f"{int(result_row['iterations'])}")
        m5.metric("Sections N", f"{int(result_row['N'])}")
        m6.metric("Processus MPI", f"{int(result_row['P'])}")
        m7.metric("Allongement", f"{result_row['AR']:.3f}")
        st.dataframe(metrics, **STRETCH)

        phase_columns = [
            "t_calcul_local", "t_calcul_distant", "t_attente",
            "t_halo", "t_reductions",
        ]
        phase_labels = {
            "t_calcul_local": "Calcul local",
            "t_calcul_distant": "Calcul distant",
            "t_attente": "Attente MPI",
            "t_halo": "Halo MPI",
            "t_reductions": "Réductions MPI",
        }
        phase_df = pd.DataFrame({
            "Phase": [phase_labels[column] for column in phase_columns],
            "Temps (s)": [float(result_row[column]) for column in phase_columns],
        })
        st.plotly_chart(
            go.Figure(go.Bar(x=phase_df["Phase"], y=phase_df["Temps (s)"],
                             marker_color="#7B3D0B"))
            .update_layout(title="Métrologie MPI — pire temps par phase",
                           yaxis_title="Temps (s)", height=360),
            **STRETCH)
        profile = execution["profile"]
        if profile is not None:
            st.plotly_chart(fortran_profile_figure(profile), **STRETCH)
            st.dataframe(profile, **STRETCH)
        else:
            st.info("Aucun profil Γ(y) n'a été demandé ou produit par Fortran.")
        gradient = execution["gradient"]
        if gradient is not None:
            st.subheader("Échanges des gradients de circulation MPI")
            st.caption(
                "Fortran représente Γ/V∞ comme une constante par section. "
                "Le gradient échangé est donc le saut entre deux sections, "
                "avec des sauts de bord g₁ et −gₙ."
            )
            gradient_columns = [
                "noeud", "y_noeud_m", "gradient_saut_gamma_sur_V",
                "gradient_gamma_sur_V_par_m", "rang_proprietaire",
            ]
            st.dataframe(gradient[gradient_columns], hide_index=True, **STRETCH)
            st.plotly_chart(fortran_gradient_figure(gradient), **STRETCH)
        else:
            st.info("Aucun gradient de circulation n'a été produit par Fortran.")
        st.caption(
            f"Processus MPI : {int(result_row['P'])} | "
            f"Temps total Python de lancement : {execution['duration']:.3f} s | "
            f"Temps de compilation : {compilation['duration']:.3f} s"
        )
        with st.expander("Journal de compilation Fortran"):
            st.code(compilation["output"], language="text")
        with st.expander("Sortie du solveur Fortran/MPI"):
            st.code(execution["stdout"], language="text")
            if execution["stderr"]:
                st.code(execution["stderr"], language="text")

    st.divider()
    st.subheader("Comparaison de scénarios et convergence du maillage 1D")
    st.caption(
        "Chaque valeur de N est le nombre de sections de la ligne portante 1D "
        "résolue par Fortran. La campagne compare les résultats Fortran aux "
        "références analytiques sans remplacer le calcul scientifique."
    )
    ccmp1, ccmp2, ccmp3 = st.columns(3)
    with ccmp1:
        comparison_geometry = st.selectbox(
            "Géométrie de la campagne",
            ["Trapézoïdale", "Elliptique"],
            key="comparison_geometry",
        )
    with ccmp2:
        comparison_incidence = st.multiselect(
            "Incidences (°)",
            [-4.0, 0.0, 2.0, 5.0, 8.0, 10.0],
            default=[0.0, 5.0, 10.0],
            key="comparison_incidence",
        )
    with ccmp3:
        comparison_meshes = st.multiselect(
            "Maillages 1D N (sections)",
            [8, 16, 24, 32, 48, 64, 96, 128, 256],
            default=[16, 32, 64],
            key="comparison_meshes",
        )
    comparison_processes = st.multiselect(
        "Processus MPI à comparer",
        [1, 2, 4, 8],
        default=[1, 2],
        key="comparison_processes",
    )
    launch_comparison = st.button(
        "Compiler et lancer la campagne de comparaison",
        type="primary",
        key="launch_comparison",
    )
    if launch_comparison:
        if not comparison_incidence or not comparison_meshes or not comparison_processes:
            st.error("Sélectionnez au moins une incidence, un maillage 1D et un nombre de processus.")
        else:
            geometry_flag = int(comparison_geometry == "Elliptique")
            geometry_name = "elliptique" if geometry_flag else "trapézoïdale"
            campaign_rows: list[dict[str, object]] = []
            try:
                with st.spinner("Compilation du solveur Fortran MPI pour la campagne…"):
                    compile_solver()
                total_cases = len(comparison_incidence) * len(comparison_meshes) * len(comparison_processes)
                progress = st.progress(0.0)
                completed = 0
                comparison_cache = st.session_state.setdefault(
                    "fortran_comparison_cache", {})
                for incidence_case in comparison_incidence:
                    for mesh_case in comparison_meshes:
                        mesh_effective = int(mesh_case) + int(mesh_case) % 2
                        for process_case in comparison_processes:
                            cache_key = (
                                geometry_name, float(span), float(root_chord),
                                float(tip_chord), float(incidence_case),
                                mesh_effective, int(process_case),
                                float(fortran_alpha_l0), float(fortran_a0),
                                float(fortran_vinf), float(rho),
                                float(fortran_tol), int(fortran_itmax),
                                int(fortran_overlap),
                            )
                            if cache_key in comparison_cache:
                                campaign_rows.append(comparison_cache[cache_key].copy())
                                completed += 1
                                progress.progress(completed / total_cases)
                                continue
                            parameters_case = {
                                "n": mesh_effective,
                                "envergure": float(span),
                                "croot": float(root_chord),
                                "ctip": float(tip_chord),
                                "incidence": float(incidence_case),
                                "vrillage": 0.0,
                                "alpha_l0": float(fortran_alpha_l0),
                                "a0": float(fortran_a0),
                                "elliptique": geometry_flag,
                                "vinf": fortran_vinf,
                                "rho": float(rho),
                                "tol": float(fortran_tol),
                                "itmax": int(fortran_itmax),
                                "rep": 1,
                                "recouvrement": int(fortran_overlap),
                            }
                            try:
                                execution_case = run_solver(
                                    parameters_case, int(process_case), dump_profile=True)
                                row = execution_case["metrics"].iloc[0].to_dict()
                                reference = analytical_reference(
                                    geometry_name, float(span), float(root_chord),
                                    float(tip_chord), float(incidence_case), 0.0,
                                    float(fortran_alpha_l0), float(fortran_a0))
                                row.update({
                                    "géométrie": geometry_name,
                                    "incidence_deg": float(incidence_case),
                                    "N_demande": int(mesh_case),
                                    "N_effectif": mesh_effective,
                                    "P": int(process_case),
                                    **reference,
                                    "erreur_CL_pct": 100.0 * abs(
                                        (float(row["CL"]) - reference["CL_ref"]) /
                                        max(abs(reference["CL_ref"]), 1e-14)),
                                    "erreur_CDi_pct": 100.0 * abs(
                                        (float(row["CDi"]) - reference["CDi_ref"]) /
                                        max(abs(reference["CDi_ref"]), 1e-14)),
                                    "profil": execution_case["profile"],
                                    "gradient": execution_case["gradient"],
                                    "statut": "OK",
                                })
                            except FortranBridgeError as exc:
                                row = {
                                    "géométrie": geometry_name,
                                    "incidence_deg": float(incidence_case),
                                    "N_demande": int(mesh_case),
                                    "N_effectif": mesh_effective,
                                    "P": int(process_case),
                                    "statut": f"ERREUR: {exc}",
                                }
                            comparison_cache[cache_key] = row.copy()
                            campaign_rows.append(row)
                            completed += 1
                            progress.progress(completed / total_cases)
                st.session_state["fortran_comparison"] = campaign_rows
                st.session_state.pop("fortran_comparison_error", None)
            except FortranBridgeError as exc:
                st.session_state["fortran_comparison_error"] = str(exc)
                st.session_state.pop("fortran_comparison", None)

    if "fortran_comparison_error" in st.session_state:
        st.error("Campagne Fortran impossible")
        st.code(st.session_state["fortran_comparison_error"], language="text")
    if "fortran_comparison" in st.session_state:
        campaign_rows = st.session_state["fortran_comparison"]
        campaign_df = pd.DataFrame(campaign_rows)
        st.subheader("Tableau comparatif")
        visible_columns = [
            "statut", "géométrie", "incidence_deg", "N_demande", "N_effectif",
            "P", "CL", "CL_ref", "erreur_CL_pct", "CDi", "CDi_ref",
            "erreur_CDi_pct", "AR", "AR_ref", "t_min",
        ]
        st.dataframe(
            campaign_df[[column for column in visible_columns if column in campaign_df]],
            hide_index=True, **STRETCH)
        valid_df = campaign_df[campaign_df["statut"] == "OK"].copy()
        if not valid_df.empty:
            chart1, chart2 = st.columns(2)
            with chart1:
                fig_cl = go.Figure()
                for label, group in valid_df.groupby("géométrie"):
                    fig_cl.add_trace(go.Scatter(
                        x=group["incidence_deg"], y=group["CL"],
                        mode="lines+markers", name=f"Fortran — {label}"))
                    fig_cl.add_trace(go.Scatter(
                        x=group["incidence_deg"], y=group["CL_ref"],
                        mode="lines+markers", name=f"Analytique — {label}",
                        line={"dash": "dash"}))
                st.plotly_chart(
                    fig_cl.update_layout(title="CL : Fortran / analytique",
                                         xaxis_title="Incidence (°)", yaxis_title="CL"),
                    **STRETCH)
            with chart2:
                fig_error = go.Figure()
                for label, group in valid_df.groupby("géométrie"):
                    grouped = group.groupby("N_effectif", as_index=False)["erreur_CL_pct"].mean()
                    fig_error.add_trace(go.Scatter(
                        x=grouped["N_effectif"], y=grouped["erreur_CL_pct"],
                        mode="lines+markers", name=label))
                st.plotly_chart(
                    fig_error.update_layout(
                        title="Convergence du maillage 1D",
                        xaxis_title="N : sections Fortran",
                        yaxis_title="Erreur moyenne CL (%)"),
                    **STRETCH)