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

def tracer1D_comparaison(t, data1, data2):
    """
    Compare la vitesses, pression et énergie des deux solutions à un temps t fixé
    :param t: int, indice entre 0 et N
    :param data1: Donnee1D, regroupe l'ensemble des données du problème
    :param data2: Donnee1D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(data1.x, data1.U[t, :, 0], 'b-', lw=2, label=data1.label)
    ax1.plot(data2.x, data2.U[t, :, 0], marker = '.',mec = 'blue', mew = 2, ms = 9,mfc = 'none', lw=0, linestyle='', label=data2.label)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(data1.x, data1.U[t, :, 1], 'r-', lw=2, label=data1.label)
    ax2.plot(data2.x, data2.U[t, :, 1], marker = '.',mec = 'red', mew = 2, ms = 9,mfc = 'none', lw=0, linestyle='', label=data2.label)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    plt.show()

def anim1D_comparaison(data1, data2):
    """
    Compare l'évolution de la vitesse, la pression et l'énergie avec deux solutions différentes
    :param data1: Donnee1D, regroupe l'ensemble des données du problème avec la solution 1
    :param data2: Donnee1D, regroupe l'ensemble des données du problème avec la solution 2
    :return: plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    line1_num, = ax1.plot([], [], color='blue', lw=2, linestyle='-', label=data1.label)
    line1_ana, = ax1.plot([], [], color='blue', lw=2, linestyle='--', label=data2.label)
    ax1.set_xlim(data1.x[0], data1.x[-1])
    ax1.set_ylim(np.min(data1.U[:, :, 0]) * 1.1, np.max(data1.U[:, :, 0]) * 1.1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Vitesse v (m/s)')
    ax1.set_title('Champ des vitesses')
    ax1.legend()
    ax1.grid(True)

    line2_num, = ax2.plot([], [], color='red', lw=2, linestyle='-', label=data1.label)
    line2_ana, = ax2.plot([], [], color='red', lw=2, linestyle='--', label=data2.label)
    ax2.set_xlim(data1.x[0], data1.x[-1])
    ax2.set_ylim(np.min(data1.U[:, :, 1]) * 1.1, np.max(data1.U[:, :, 1]) * 1.1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Pression p (Pa)')
    ax2.set_title('Champ des pressions')
    ax2.legend()
    ax2.grid(True)

    line3_num, = ax3.plot([], [], color='orange', lw=2, linestyle='-', label=data1.label)
    line3_ana, = ax3.plot([], [], color='orange', lw=2, linestyle='--', label=data2.label)
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
        title.set_text(data1.label + " VS " + data2.label)
        return line1_num, line1_ana, line2_num, line2_ana, line3_num, line3_ana

    def update(n):
        line1_num.set_data(data1.x, data1.U[n, :, 0])
        line1_ana.set_data(data1.x, data2.U[n, :, 0])
        line2_num.set_data(data1.x, data1.U[n, :, 1])
        line2_ana.set_data(data1.x, data2.U[n, :, 1])
        line3_num.set_data(data1.t[:n + 1], data1.E[:n + 1])
        line3_ana.set_data(data1.t[:n + 1], data2.E[:n + 1])
        return line1_num, line1_ana, line2_num, line2_ana, line3_num, line3_ana

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    anim = FuncAnimation(fig, update, init_func=init, frames=data1.N, interval=20, blit=True)
    plt.show()
    return anim

def erreur1D_trace(eps, M_li = np.array([100, 200, 400, 800, 1600]), xc = (0,400), f = "LW"):
    c1 = "blue"
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


def anim2D(data):
    """
    Trace l'évolution de la vitesse (vx, vy), la pression et l'énergie des données de data en 2D
    :param data: Donnee2D, regroupe l'ensemble des données du problème
    :return: plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]

    extent = [data.x[0], data.x[-1], data.y[0], data.y[-1]]

    im1 = ax1.imshow(np.zeros((len(data.y), len(data.x))), cmap='jet', origin='lower', extent=extent,
                     vmin=np.min(data.U[..., 0]), vmax=np.max(data.U[..., 0]))
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Position y (m)')
    ax1.set_title('Champ des vitesses vx')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.zeros((len(data.y), len(data.x))), cmap='jet', origin='lower', extent=extent,
                     vmin=np.min(data.U[..., 1]), vmax=np.max(data.U[..., 1]))
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Position y (m)')
    ax2.set_title('Champ des vitesses vy')
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(np.zeros((len(data.y), len(data.x))), cmap='viridis', origin='lower', extent=extent,
                     vmin=np.min(data.U[..., 2]), vmax=np.max(data.U[..., 2]))
    ax3.set_xlabel('Position x (m)')
    ax3.set_ylabel('Position y (m)')
    ax3.set_title('Champ des pressions')
    fig.colorbar(im3, ax=ax3)

    kinetic = 0.5 * data.rho * (data.U[..., 0] ** 2 + data.U[..., 1] ** 2)
    potential = data.U[..., 2] / (data.rho * data.c ** 2)
    data.E = np.sum(kinetic + potential, axis=(1, 2))

    line4, = ax4.plot([], [], color='orange', lw=2)
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
        im1.set_data(data.U[n, :, :, 0])
        im2.set_data(data.U[n, :, :, 1])
        im3.set_data(data.U[n, :, :, 2])
        line4.set_data(data.t[:n + 1], data.E[:n + 1])

        title.set_text(f'Évolution des champs à t = {data.t[n]:.3f} s')
        return im1, im2, im3, line4

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax4.set_box_aspect(1)

    plt.tight_layout()
    anim = FuncAnimation(fig, update, init_func=init, frames=data.N, interval=5, blit=True)
    plt.show()
    return anim