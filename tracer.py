import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def anim1D(data):
    """
    Trace l'évolution de la vitesse, la pression et l'énergie des données de data
    :param data: Donnee1D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    line1, = ax1.plot([], [], color='blue', lw=2)
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

    line3, = ax3.plot([], [], color='orange', lw=2)
    data.E = np.array([sum([0.5 * data.rho * data.U[n, i, 0] ** 2 + data.U[n, i, 1]/(data.rho*data.c**2) for i in range(data.M)]) for n in range(0, data.N)])
    ax3.set_xlim(data.t[0], data.t[-1])
    ax3.set_ylim(np.min(data.E)*1.1, np.max(data.E)*1.1)
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
        title.set_text('')
        return line1, line2, line3

    def update(n):
        line1.set_data(data.x, data.U[n, :, 0])
        line2.set_data(data.x, data.U[n, :, 1])
        line3.set_data(data.t[:n+1], data.E[:n+1])
        title.set_text(f"Temps t = {n * data.dt:.4f} s")
        return line1, line2, line3

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=20, blit=True)
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

def tracer1D_analytiqueVSnumerique(t, data1, data2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(data1.x, data1.U[t, :, 0], 'b-', lw=2, label='LaxWendroff')
    ax1.plot(data2.x, data2.U[t, :, 0], 'b--', lw=2, label='Analytique')
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(data1.x, data1.U[t, :, 1], 'r-', lw=2, label='LaxWendroff')
    ax2.plot(data2.x, data2.U[t, :, 1], 'r--', lw=2, label='Analytique')
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    plt.show()

def anim1D_analytiqueVSnumerique(data1, data2):
    """
    Compare l'évolution de la vitesse, la pression et l'énergie avec Lax-Wendroff et la solution analytique
    :param data1: Donnee1D, regroupe l'ensemble des données du problème avec Lax Wendroff
    :param data2: Donnee1D, regroupe l'ensemble des données du problème avec la solution analytique
    :return: plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    line1_num, = ax1.plot([], [], color='blue', lw=2, linestyle='-', label='LaxWendroff')
    line1_ana, = ax1.plot([], [], color='blue', lw=2, linestyle='--', label='Analytique')
    ax1.set_xlim(data1.x[0], data1.x[-1])
    ax1.set_ylim(np.min(data1.U[:, :, 0]) * 1.1, np.max(data1.U[:, :, 0]) * 1.1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    line2_num, = ax2.plot([], [], color='red', lw=2, linestyle='-', label='LaxWendroff')
    line2_ana, = ax2.plot([], [], color='red', lw=2, linestyle='--', label='Analytique')
    ax2.set_xlim(data1.x[0], data1.x[-1])
    ax2.set_ylim(np.min(data1.U[:, :, 1]) * 1.1, np.max(data1.U[:, :, 1]) * 1.1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    line3_num, = ax3.plot([], [], color='orange', lw=2, linestyle='-', label='LaxWendroff')
    line3_ana, = ax3.plot([], [], color='orange', lw=2, linestyle='--', label='Analytique')
    data1.E = np.array([sum([0.5 * data1.rho * data1.U[n, i, 0] ** 2 + data1.U[n, i, 1] / (data1.rho * data1.c ** 2) for i in range(data1.M)]) for n in range(data1.N)])
    data2.E = np.array([sum([0.5 * data1.rho * data2.U[n, i, 0] ** 2 + data2.U[n, i, 1] / (data1.rho * data1.c ** 2) for i in range(data1.M)]) for n in range(data1.N)])
    ax3.set_xlim(data1.t[0], data1.t[-1])
    ax3.set_ylim(np.min(data1.E)*0.9, np.max(data1.E)*1.1)
    ax3.set_xlabel('Temps t (s)')
    ax3.set_ylabel('Energie (J)')
    ax3.set_title("Evolution de l'energie (J)")
    ax3.legend()
    ax3.grid(True)

    title = fig.suptitle('', fontsize=14)

    def init():
        line1_num.set_data([], [])
        line1_ana.set_data([], [])
        line2_num.set_data([], [])
        line2_ana.set_data([], [])
        line3_num.set_data([], [])
        line3_ana.set_data([], [])
        title.set_text('')
        return line1_num, line1_ana, line2_num, line2_ana, line3_num, line3_ana

    def update(n):
        line1_num.set_data(data1.x, data1.U[n, :, 0])
        line1_ana.set_data(data1.x, data2.U[n, :, 0])
        line2_num.set_data(data1.x, data1.U[n, :, 1])
        line2_ana.set_data(data1.x, data2.U[n, :, 1])
        line3_num.set_data(data1.t[:n + 1], data1.E[:n + 1])
        line3_ana.set_data(data1.t[:n + 1], data2.E[:n + 1])
        title.set_text(f"Temps t = {n * data1.dt:.4f} s")
        return line1_num, line1_ana, line2_num, line2_ana, line3_num, line3_ana

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    anim = FuncAnimation(fig, update, init_func=init, frames=data1.N, interval=20, blit=True)
    plt.show()
    return anim

def erreur1D_trace(eps, M_li = np.array([100, 200, 400, 800, 1600]), xc = (0,400)):
    def trace(eps):
        dx_li = xc[1]/M_li
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(np.log10(dx_li), np.log10(eps[:,0]), 'b.-', lw=2, ms = 8)
        ax1.grid(True)
        ax1.set_xlabel('dx (en m)')
        ax1.set_ylabel('log(erreur)')
        ax1.set_title("Erreur en vitesse")

        ax2.plot(np.log10(dx_li), np.log10(eps[:,1]), 'r.-', lw=2, ms = 8)
        ax2.grid(True)
        ax2.set_xlabel('dx (en m)')
        ax2.set_ylabel('log(erreur)')
        ax2.set_title("Erreur en pression")

        print("Degré d'erreur en vitesse",np.polyfit(np.log10(dx_li), np.log10(eps[:,0]), 1))
        print("Degré d'erreur en pression", np.polyfit(np.log10(dx_li), np.log10(eps[:,1]), 1))
    if len(eps.shape) == 2:
        trace(eps)

    plt.show()