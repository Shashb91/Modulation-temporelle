import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from matplotlib.widgets import Slider

def anim1D(data, **kwargs):
    """
    Trace l'évolution de la vitesse, la pression et l'énergie des données de data
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    line1, = ax1.plot([], [], color='darkblue', lw=2)
    ax1.set_xlim(data.x[0], data.x[-1])
    ax1.set_ylim(np.min(data.U[:, :, 0]) * 1.1, np.max(data.U[:, :, 0]) * 1.1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.grid(True)

    line2, = ax2.plot([], [], color='red', lw=2)
    ax2.set_xlim(data.x[0], data.x[-1])
    ax2.set_ylim(np.min(data.U[:, :, 1]) * 1.1, np.max(data.U[:, :, 1]) * 1.1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.grid(True)

    line3, = ax3.plot([], [], color='gold', lw=2)
    ax3.set_xlim(data.t[0], data.t[-1])
    ax3.set_ylim(0, np.max(data.E)*1.1)
    ax3.set_xlabel('Position x (m)')
    ax3.set_xlabel('Temps t (s)')
    ax3.set_ylabel('Energie (J)')
    ax3.set_title("Evolution de l'energie (J)")
    ax3.grid(True)

    title = fig.suptitle('', fontsize=14)
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        title.set_text('Courbe de pression, vitesse et énergie pour la solution '+data.label)
        return line1, line2, line3

    def update(n):
        line1.set_data(data.x, data.U[n, :, 0])
        line2.set_data(data.x, data.U[n, :, 1])
        line3.set_data(data.t[:n+1], data.E[:n+1])
        return line1, line2, line3

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)

    if "interval" in kwargs.keys(): interval = kwargs["interval"]
    else: interval = 5
    anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=interval, blit=True)
    plt.show()
    return anim

def tracer1D(data,t):
    """
    Trace la vitesse et la pression selon x à un temps t fixé
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :param t: int, indice entre 0 et N
    :return: plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(data.x, data.U[t, :, 0], 'b-', lw=2)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.grid(True)

    ax2.plot(data.x, data.U[t, :, 1], 'r-', lw=2)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.grid(True)

    plt.show()
    
def anim1D_comparaison(data1, data2, **kwargs):
    """
    Compare l'évolution de la vitesse, la pression et l'énergie avec deux solutions différentes
    :param data1: Donnee1D, regroupe l'ensemble des données du problème avec la solution 1
    :param data2: Donnee1D, regroupe l'ensemble des données du problème avec la solution 2
    :return: plot
    """
    assert data1 == data2, "Les paramètres sont différents"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    line1_1, = ax1.plot([], [], color='darkblue', lw=2, linestyle='-', label=data1.label)
    line1_2, = ax1.plot([], [], color='dodgerblue', lw=2, linestyle='--', label=data2.label)
    ax1.set_xlim(data1.x[0], data1.x[-1])
    ax1.set_ylim(np.min(data1.U[:, :, 0]) * 1.1, np.max(data1.U[:, :, 0]) * 1.1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    line2_1, = ax2.plot([], [], color='red', lw=2, linestyle='-', label=data1.label)
    line2_2, = ax2.plot([], [], color='deeppink', lw=2, linestyle='--', label=data2.label)
    ax2.set_xlim(data1.x[0], data1.x[-1])
    ax2.set_ylim(np.min(data1.U[:, :, 1]) * 1.1, np.max(data1.U[:, :, 1]) * 1.1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    line3_1, = ax3.plot([], [], color='gold', lw=2, linestyle='-', label=data1.label)
    line3_2, = ax3.plot([], [], color='darkorange', lw=2, linestyle='--', label=data2.label)
    ax3.set_xlim(data1.t[0], data1.t[-1])
    ax3.set_ylim(0, max(np.max(data1.E), np.max(data2.E))*1.1)
    ax3.set_xlabel('Temps t (s)')
    ax3.set_ylabel('Energie (J)')
    ax3.set_title("Evolution de l'energie (J)")
    ax3.legend()
    ax3.grid(True)

    title = fig.suptitle('', fontsize=14)

    def init():
        line1_1.set_data([], [])
        line1_2.set_data([], [])
        line2_1.set_data([], [])
        line2_2.set_data([], [])
        line3_1.set_data([], [])
        line3_2.set_data([], [])
        title.set_text(data1.label + " VS " + data2.label)
        return line1_1, line1_2, line2_1, line2_2, line3_1, line3_2

    def update(n):
        line1_1.set_data(data1.x, data1.U[n, :, 0])
        line1_2.set_data(data1.x, data2.U[n, :, 0])
        line2_1.set_data(data1.x, data1.U[n, :, 1])
        line2_2.set_data(data1.x, data2.U[n, :, 1])
        line3_1.set_data(data1.t[:n + 1], data1.E[:n + 1])
        line3_2.set_data(data1.t[:n + 1], data2.E[:n + 1])
        return line1_1, line1_2, line2_1, line2_2, line3_1, line3_2

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)

    if "interval" in kwargs.keys(): interval = kwargs["interval"]
    else: interval = 20
    anim = FuncAnimation(fig, update, init_func=init, frames=data1.N, interval=interval, blit=True)
    plt.show()
    return anim

def tracer1D_comparaison(t, data1, data2):
    """
    Compare la vitesse, pression et énergie des deux solutions à un temps t fixé
    :param t: int, indice entre 0 et N
    :param data1: Donnee1D, regroupe l'ensemble des données du problème tracé en trait plein
    :param data2: Donnee1D, regroupe l'ensemble des données du problème tracé avec des markers transparents
    :return: plot
    """
    assert data1 == data2, "Les paramètres sont différents"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(data1.x, data1.U[t, :, 0], 'b-', lw=2, label=data1.label)
    ax1.plot(data2.x, data2.U[t, :, 0], marker = '.',mec = 'dodgerblue', mew = 2, ms = 9,mfc = 'none', lw=0, linestyle='', label=data2.label)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(data1.x, data1.U[t, :, 1], 'r-', lw=2, label=data1.label)
    ax2.plot(data2.x, data2.U[t, :, 1], marker = '.',mec = 'deeppink', mew = 2, ms = 9,mfc = 'none', lw=0, linestyle='', label=data2.label)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    plt.show()

def erreur_trace(eps, M_li = np.array([100, 200, 400, 800, 1600]), xc = (0,30), f = "LW"):
    c1 = "darkblue"
    c2 = "red"

    dx_li = xc[1]/M_li
    fig, (ax1, ax2) = plt.subplots(1,2)
    title = fig.suptitle('', fontsize=14)

    ax1.plot(np.log10(dx_li), np.log10(eps[:,0]),color = c1, linestyle = '-', marker = ".", lw=2, ms = 8)
    ax1.grid(True)
    ax1.set_xlabel('log(dx) (en m)')
    ax1.set_ylabel('log(erreur)')
    ax1.set_title("Erreur en vitesse")

    ax2.plot(np.log10(dx_li), np.log10(eps[:,1]),color = c2, linestyle = '-', marker = ".", lw=2, ms = 8)
    ax2.grid(True)
    ax2.set_xlabel('log(dx) (en m)')
    ax2.set_ylabel('log(erreur)')
    ax2.set_title("Erreur en pression")

    title.set_text("Erreur avec " + f)
    print("Degré d'erreur en vitesse",np.polyfit(np.log10(dx_li), np.log10(eps[:,0]), 1)[0])
    print("Degré d'erreur en pression", np.polyfit(np.log10(dx_li), np.log10(eps[:,1]), 1)[0])
    plt.show()

"""
Fonctions adpatées pour les instances de Donnee2D uniquement !
Pas de fonctions tracer2D ni de fonctions tracer2D_comparaison : 
tracé selon x et y en image, fixé avec un subplot ... bcp d'efforts, pas forcément utile
"""

def anim2D(data, **kwargs):
    """
    Trace l'évolution de la vitesse (vx, vy), la pression et l'énergie des données de data en 2D
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]

    extent = [data.x[0], data.x[-1], data.y[0], data.y[-1]]
    param = data.N//2
    v_min, v_max = min(np.min(data.U[param:, ...,0]),np.min(data.U[param:,...,1])), max(np.max(data.U[param:,...,0]),np.max(data.U[param:,...,1]))
    im1 = ax1.imshow(np.zeros((len(data.y), len(data.x))), cmap='seismic', origin='lower', extent=extent, norm = colors.TwoSlopeNorm(vmin=v_min, vmax=v_max, vcenter = 0))
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Position y (m)')
    ax1.set_title('Champ des vitesses vx')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.zeros((len(data.y), len(data.x))), cmap='seismic', origin='lower', extent=extent,norm = colors.TwoSlopeNorm(vmin=v_min, vmax=v_max, vcenter = 0))
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Position y (m)')
    ax2.set_title('Champ des vitesses vy')
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(np.zeros((len(data.y), len(data.x))), cmap='seismic', origin='lower', extent=extent,norm = colors.TwoSlopeNorm(vmin=np.min(data.U[param:,..., 2]), vmax=np.max(data.U[param:,..., 2]), vcenter = 0))
    ax3.set_xlabel('Position x (m)')
    ax3.set_ylabel('Position y (m)')
    ax3.set_title('Champ des pressions')
    fig.colorbar(im3, ax=ax3)

    line4, = ax4.plot([], [], color='gold', lw=2)
    ax4.set_xlim(data.t[0], data.t[-1])
    ax4.set_ylim(np.min(data.E) * 0.9, np.max(data.E) * 1.1)
    ax4.set_xlabel('Temps t (s)')
    ax4.set_ylabel('Energie (J)')
    ax4.set_title("Evolution de l'energie (J)")
    ax4.grid(True)
    title = fig.suptitle('', fontsize=14)

    def init():
        im1.set_data(np.zeros((len(data.y), len(data.x))))
        im2.set_data(np.zeros((len(data.y), len(data.x))))
        im3.set_data(np.zeros((len(data.y), len(data.x))))
        line4.set_data([], [])
        title.set_text('Courbes 2D pour la solution ' + data.label)
        return im1, im2, im3, line4

    def update(n):
        im1.set_data(data.U[n, :, :, 0].transpose())
        im2.set_data(data.U[n, :, :, 1].transpose())
        im3.set_data(data.U[n, :, :, 2].transpose())
        line4.set_data(data.t[:n + 1], data.E[:n + 1])

        title.set_text(f'Évolution des champs à t = {data.t[n]:.3f} s')
        return im1, im2, im3, line4

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax4.set_box_aspect(1)

    plt.tight_layout()
    if "interval" in kwargs.keys(): interval = kwargs["interval"]
    else: interval = 30 * (not data.label.endswith("compressé")) + int(data.label.endswith("compressé"))
    anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=interval, blit=True)
    plt.show()
    return anim

def tracer2D(data,t, y):
    """
    Trace la vitesse et la pression selon x à un temps t fixé
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :param t: int, indice entre 0 et N
    :return: plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    ax1.plot(data.x, data.U[t, :, y, 0], 'b-', lw=2)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.grid(True)

    ax2.plot(data.x, data.U[t, :, y, 1], 'g-', lw=2)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Vitesse v (m/s)')
    ax2.set_title('Champ des vitesses en y')
    ax2.grid(True)

    ax3.plot(data.x, data.U[t, :, y, 2], 'r-', lw=2)
    ax3.set_xlabel('Position x (m)')
    ax3.set_ylabel('Pression p (Pa)')
    ax3.set_title('Champ des pressions')
    ax3.grid(True)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    plt.show()

"""
Fonctions de tracé pour la modulation temporelle, avec l'impédance
"""

def tracer_mt(data):
    """
    Trace la masse volumique et le module d'Young modulés en temps
    :param data: Donnee1D
    :return: plot
    """
    fig, (axs) = plt.subplots(2, 2)
    ax1, ax2 = axs[0,0], axs[0,1]
    ax3, ax4 = axs[1,0], axs[1,1]
    r, e, c, Z = [], [], [], []
    for i in range(data.N):
        r.append(data.rho_mt(data)(data.dt * i)[0])
        e.append(data.E_mt(data)(data.dt * i)[0])
        c.append(np.sqrt(data.E_mt(data)(data.dt * i)[0]/data.rho_mt(data)(data.dt * i)[0]))
        Z.append(data.rho_mt(data)(data.dt * i)[0]*np.sqrt(data.E_mt(data)(data.dt * i)[0]/data.rho_mt(data)(data.dt * i)[0]))

    ax1.plot(data.t, r, lw=2, c='red')
    ax1.grid(True)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(f"Masse volumique $(g.m^{-3})$ ")

    ax2.plot(data.t, e, lw=2, c='darkblue')
    ax2.grid(True)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("Module d'Young (Pa)")

    ax3.plot(data.t, c, lw=2, c='gold')
    ax3.grid(True)
    ax3.set_xlabel('t (s)')
    ax3.set_ylabel(f'Célérité $(m.s^{-1})$')

    ax4.plot(data.t, Z, lw=2, c='darkorange')
    ax4.grid(True)
    ax4.set_xlabel('t (s)')
    ax4.set_ylabel(f'Impédance $(kg.m^{-2}.s^{-1})$')

    plt.show()

def anim1D_mt(data, **kwargs):
    """
    Trace l'évolution de la vitesse, la pression et l'énergie des données de data dans le cas d'un milieu modulé en temps
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, (axs) = plt.subplots(2, 2, figsize=(12, 5))
    ax1, ax2 = axs[0, 0], axs[1, 0]
    ax3, ax4 = axs[0, 1], axs[1, 1]
    rho = [data.rho_mt(data)(data.dt * n)[0] for n in range(data.N)]
    E = [data.E_mt(data)(data.dt * n)[0] for n in range(data.N)]

    line1, = ax1.plot([], [], color='darkblue', lw=2)
    ax1.set_xlim(data.x[0], data.x[-1])
    ax1.set_ylim(np.min(data.U[:, :, 0]) * 1.1, np.max(data.U[:, :, 0]) * 1.1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.grid(True)

    line2, = ax2.plot([], [], color='red', lw=2)
    ax2.set_xlim(data.x[0], data.x[-1])
    ax2.set_ylim(np.min(data.U[:, :, 1]) * 1.1, np.max(data.U[:, :, 1]) * 1.1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.grid(True)

    line3, = ax3.plot([], [], color='gold', lw=2)
    ax3.set_xlim(data.t[0], data.t[-1])
    ax3.set_ylim(0 , np.max(data.E) * 1.1)
    ax3.set_xlabel('Temps t (s)')
    ax3.set_ylabel('Energie (J)')
    ax3.set_title("Evolution de l'energie (J)")
    ax3.grid(True)

    line4, = ax4.plot([], [], color='green', lw=2)
    Z = np.array([np.sqrt(rho[n]*E[n]) for n in range(data.N)])
    ax4.set_xlim(data.t[0], data.t[-1])
    ax4.set_ylim(np.min(Z) * 0.9, np.max(Z) * 1.1)
    ax4.set_xlabel('Temps t (s)')
    ax4.set_ylabel('Impédance  $(kg.m^{-2}.s^{-1})$')
    ax4.set_title("Evolution de l'impédance (J)")
    ax4.grid(True)

    title = fig.suptitle('', fontsize=14)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        title.set_text("Courbe de pression, vitesse, énergie et d'impédance pour la solution " + data.label)
        return line1, line2, line3, line4

    def update(n):
        line1.set_data(data.x, data.U[n, :, 0])
        line2.set_data(data.x, data.U[n, :, 1])
        line3.set_data(data.t[:n + 1], data.E[:n + 1])
        line4.set_data(data.t[:n + 1], Z[:n + 1])
        return line1, line2, line3, line4


    if "interval" in kwargs.keys(): interval = kwargs["interval"]
    else: interval = 5
    anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=interval, blit=True)
    plt.show()
    return anim

def anim1D_mt_comparaison(data1, data2, **kwargs):
    """
    Trace la comparaison entre deux Donnee2D, l'évolution de la vitesse, la pression et l'énergie des données de data1 et data2 dans le cas d'un milieu modulé en temps
    ATTENTION : LA MODULATION DOIT ÊTRE IDENTIQUE !!!
    :param data1: Donnee2D, regroupe l'ensemble des données du problème
    :param data2: Donnee2D, regroupe l'ensemble des données du problème
    :return: plot
    """
    assert data1 == data2, "Les paramètres sont différents"
    fig, (axs) = plt.subplots(2, 2, figsize=(12, 5))
    ax1, ax2 = axs[0, 0], axs[1, 0]
    ax3, ax4 = axs[0, 1], axs[1, 1]
    rho = [data1.rho_mt(data1)(data1.dt * n)[0] for n in range(data1.N)]
    E = [data1.E_mt(data1)(data1.dt * n)[0] for n in range(data1.N)]

    line1_1, = ax1.plot([], [], color='darkblue', lw=2  , label = data1.label, ls = '-')
    line1_2, = ax1.plot([], [], color='dodgerblue', lw=2, label = data2.label, ls = '--')
    ax1.set_xlim(data1.x[0], data1.x[-1])
    ax1.set_ylim(np.min(data1.U[:, :, 0]) * 1.1, np.max(data1.U[:, :, 0]) * 1.1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    line2_1, = ax2.plot([], [], color='red', lw=2, label = data1.label, ls = '-')
    line2_2, = ax2.plot([], [], color='deeppink', lw=2, label = data2.label, ls = '--')
    ax2.set_xlim(data1.x[0], data1.x[-1])
    ax2.set_ylim(np.min(data1.U[:, :, 1]) * 1.1, np.max(data1.U[:, :, 1]) * 1.1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    line3_1, = ax3.plot([], [], color='gold', lw=2, label = data1.label, ls = '-')
    line3_2, = ax3.plot([], [], color='darkorange', lw=2, label = data2.label, ls = '--')
    ax3.set_xlim(data1.t[0], data1.t[-1])
    ax3.set_ylim(0, np.max(data1.E) * 1.1)
    ax3.set_xlabel('Temps t (s)')
    ax3.set_ylabel('Energie (J)')
    ax3.set_title("Evolution de l'energie (J)")
    ax3.legend()
    ax3.grid(True)

    line4, = ax4.plot([], [], color='green', lw=2)
    Z = np.array([np.sqrt(rho[n]*E[n]) for n in range(data1.N)])
    ax4.set_xlim(data1.t[0], data1.t[-1])
    ax4.set_ylim(np.min(Z) * 0.9, np.max(Z) * 1.1)
    ax4.set_xlabel('Temps t (s)')
    ax4.set_ylabel('Impédance  $(kg.m^{-2}.s^{-1})$')
    ax4.set_title("Evolution de l'impédance (J)")
    ax4.grid(True)

    title = fig.suptitle('', fontsize=14)

    def init():
        line1_1.set_data([], [])
        line1_2.set_data([], [])
        line2_1.set_data([], [])
        line2_2.set_data([], [])
        line3_1.set_data([], [])
        line3_2.set_data([], [])
        line4.set_data([], [])
        title.set_text(data1.label + " vs " + data2.label)
        return line1_1, line2_1, line3_1, line4

    def update(n):
        line1_1.set_data(data1.x, data1.U[n, :, 0])
        line1_2.set_data(data1.x, data2.U[n, :, 0])
        line2_1.set_data(data1.x, data1.U[n, :, 1])
        line2_2.set_data(data1.x, data2.U[n, :, 1])
        line3_1.set_data(data1.t[:n + 1], data1.E[:n + 1])
        line3_2.set_data(data1.t[:n + 1], data2.E[:n + 1])
        line4.set_data(data1.t[:n + 1], Z[:n + 1])
        return line1_1, line1_2, line2_1, line2_2, line3_1, line3_2, line4


    if "interval" in kwargs.keys(): interval = kwargs["interval"]
    else: interval = 5
    anim = FuncAnimation(fig, update, init_func=init, frames=data1.N, interval=interval, blit=True)
    plt.show()
    return anim

"""
Fonctions de tracer pour l'Elasticité AntiPlane
"""

def anim2D_EAP(data, **kwargs):
    """
    Trace l'évolution de la vitesse v, les contraintes et l'énergie des données de data en 2D
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]

    extent = [data.x[0], data.x[-1], data.y[0], data.y[-1]]
    param = int(data.N/4)
    sigma_min, sigma_max = min(np.min(data.U[param:, ...,2]),np.min(data.U[param:,...,1])), max(np.max(data.U[param:,...,2]),np.max(data.U[param:,...,1]))
    im1 = ax1.imshow(np.zeros((len(data.y), len(data.x))), cmap='seismic', origin='lower', extent=extent, norm = colors.TwoSlopeNorm(vmin=sigma_min, vmax=sigma_max, vcenter = 0))
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Position y (m)')
    ax1.set_title('Champ des contraintes 11')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.zeros((len(data.y), len(data.x))), cmap='seismic', origin='lower', extent=extent,norm = colors.TwoSlopeNorm(vmin=sigma_min, vmax=sigma_max, vcenter = 0))
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Position y (m)')
    ax2.set_title('Champ des contraintes 12')
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(np.zeros((len(data.y), len(data.x))), cmap='seismic', origin='lower', extent=extent,norm = colors.TwoSlopeNorm(vmin=np.min(data.U[param:,..., 0]), vmax=np.max(data.U[param:,..., 0]), vcenter = 0))
    ax3.set_xlabel('Position x (m)')
    ax3.set_ylabel('Position y (m)')
    ax3.set_title('Champ de la vitesse')
    fig.colorbar(im3, ax=ax3)

    line4, = ax4.plot([], [], color='gold', lw=2)
    ax4.set_xlim(data.t[0], data.t[-1])
    ax4.set_ylim(np.min(data.E) * 0.9, np.max(data.E) * 1.1)
    ax4.set_xlabel('Temps t (s)')
    ax4.set_ylabel('Energie (J)')
    ax4.set_title("Evolution de l'energie (J)")
    ax4.grid(True)
    title = fig.suptitle('', fontsize=14)

    def init():
        im1.set_data(np.zeros((len(data.y), len(data.x))))
        im2.set_data(np.zeros((len(data.y), len(data.x))))
        im3.set_data(np.zeros((len(data.y), len(data.x))))
        line4.set_data([], [])
        title.set_text('Courbes 2D pour la solution ' + data.label)
        return im1, im2, im3, line4

    def update(n):
        im1.set_data(data.U[n, :, :, 1].transpose())
        im2.set_data(data.U[n, :, :, 2].transpose())
        im3.set_data(data.U[n, :, :, 0].transpose())
        line4.set_data(data.t[:n + 1], data.E[:n + 1])

        title.set_text(f'Évolution des champs à t = {data.t[n]:.3f} s')
        return im1, im2, im3, line4

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax4.set_box_aspect(1)

    plt.tight_layout()
    if "interval" in kwargs.keys(): interval = kwargs["interval"]
    else: interval = 30 * (not data.label.endswith("compressé")) + int(data.label.endswith("compressé"))
    anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=interval, blit=True)
    plt.show()
    return anim