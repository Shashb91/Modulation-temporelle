from source import*
from math import sqrt
from modulation import*
import numpy as np

class Donnee1D:
    def __init__(self, c, rho, kappa, f = 20, xc = (0, 400), tc = (0, 0.125), M = 400, CFL = 0.95, **kwargs):
        """
        Initialisation d'une instance de Donnee1D, permettant à la résolution du problème 1D de proagation en milieu homogène ou modulée en temps
        """
        self.c : float = c                                                 #célérité en m/s
        self.rho : float = rho                                             #masse volumique en g/m^3
        self.kappa : float = kappa                                                 #module d'Young en Pa
        self.f : float = f                                                 #frequence max observable
        self.xc : tuple = xc                                               #couple position min, max
        self.tc : tuple = tc                                               #couple temps init, final
        self.M : int = M                                                   #discretisation spatiale
        self.dx : float = self.xc[1]/(self.M -1)                                #infinitesimal spatial
        self.CFL: float = CFL
        self.dt : float = self.CFL * self.dx / self.c

        if "label" in kwargs.keys(): self.label = kwargs["label"]          #label
        else: self.label = ""

        self.N : int = int(self.tc[1]/self.dt)                             #discretisation temporelle
        self.x = np.linspace(self.xc[0], self.xc[1], self.M)               #axe x
        self.t = np.linspace(self.tc[0], self.tc[1], self.N)               #axe temporel

        if "U" in kwargs.keys(): self.U : np.ndarray() = kwargs["U"]       #vecteur [v, p] solution
        else: self.U : np.ndarray() = np.zeros((self.N, self.M, 2))

        self.fmax : float = self.f*10

        if "S" in kwargs.keys(): self.S = kwargs["S"]                      #donnee point source
        else: self.S = pt_source_1D

        if "xs" in kwargs.keys(): self.xs : int = kwargs["xs"]             #postion point source
        else: self.xs : int = self.M // 2

        if "E" in kwargs.keys(): self.E : np.ndarray() = kwargs["E"]       #energie
        else: self.E : np.ndarray() = np.zeros(self.N)

        if "opt" in kwargs.keys(): self.opt = kwargs["opt"]                #True si perturb sur vitesse, False si perturb sur pression
        else: self.opt : bool = True

        if "eps_r" in kwargs.keys(): self.eps_r = kwargs["eps_r"]                #Modulation temporelle
        else: self.eps_r = 0
        if "eps_E" in kwargs.keys(): self.eps_kappa = kwargs["eps_E"]
        else: self.eps_kappa = 0
        if "omega" in kwargs.keys(): self.omega = kwargs["omega"]
        else: self.omega = 0

        if "rho_mt" in kwargs.keys(): self.rho_mt = kwargs["rho_mt"]
        else: self.rho_mt = rho_sinus
        if "E_mt" in kwargs.keys(): self.kappa_mt = kwargs["E_mt"]
        else: self.kappa_mt = kappa_sinus

        if "alpha" in kwargs.keys(): self.alpha = kwargs["alpha"]
        else: self.alpha = 0.5

        if "param" in kwargs.keys(): self.param = kwargs["param"]
        else: self.param = 0.5

    def __str__(self):
        f = self.omega/(2*np.pi)
        crit = self.c/((f*4+self.f)*self.dx)
        mess = ""
        if crit <= 25:
            mess = " -> ATTENTION ! TROP PETIT, AUGEMENTER M"
        crit = str(crit) + mess
        return "\nDonnee1D(" + self.label + ", c :" + str(self.c) + ", kappa : " + str(self.kappa) + ", rho : " + str(self.rho) + ",\n" + "N : " + str(
            self.N) + ", M : " + str(self.M) + ", dx : " + str(self.dx) + ', dt : ' + str(self.dt) + ")\nlambda/dx = " + crit

    def __eq__(self, other):
        return (self.M == other.M)*(self.CFL == other.CFL)*(self.xc == other.xc)*(self.tc == other.tc)*(self.f == other.f)*(self.S == other.S)*(self.eps_r == other.eps_r)*(self.eps_kappa == other.eps_kappa)*(self.omega == other.omega)*(self.rho_mt == other.rho_mt)*(self.kappa_mt == other.kappa_mt)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __sub__(self, other):
        ecart = self.U - other.U
        return np.array([[np.max(ecart[...,0]), np.min(ecart[...,0]), np.average(ecart[...,0]), np.std(ecart[...,0])],
                         [np.max(ecart[...,1]), np.min(ecart[...,1]), np.average(ecart[...,1]), np.std(ecart[...,1])]])

    def copy(self, instance : Donnee1D):                                   #constructeur de recopie
        self.c : float = instance.c
        self.rho: float = instance.rho
        self.f: float = instance.f
        self.xc: tuple = instance.xc
        self.tc: tuple = instance.tc
        self.M: int = instance.M
        self.dx: float = instance.dx
        self.CFL: float = instance.CFL
        self.dt: float = instance.dt
        self.N : int = instance.N
        self.U : np.ndarray = instance.U
        self.fmax: float = self.f * 10
        self.opt: bool = instance.opt
        self.label: str = instance.label
        self.S = instance.S
        self.xs : list[tuple] = instance.xs
        self.E = instance.E

    def CFL_maj(self):
        """
        Permet de corriger la CFL dans un milieu modulé en temps
        :return: None
        """
        rho, E = [],[]
        for t in self.t:
            rho.append(self.rho_mt(self)(t)[0])
            E.append(self.kappa_mt(self)(t)[0])
        rho,E = np.array(rho),np.array(E)
        self.c: float = np.max(np.sqrt(E/rho))
        self.dt : float = self.CFL * self.dx / self.c
        self.N: int = int(self.tc[1] / self.dt)
        self.t = np.linspace(self.tc[0], self.tc[1], self.N)

    def calcul_energie(self):
        self.E = np.array([sum([(0.5 * self.rho * self.U[n, i, 0] ** 2 + self.U[n, i, 1] ** 2 / (2*self.rho * self.c ** 2))*self.dx for i in range(self.M)]) for n in range(0, self.N)])

class Donnee2D:
    def __init__(self, c, rho, kappa, f=20, xc=(0, 300), yc = (0, 300), tc=(0, 0.25), Mx = 150, My = 150, opt = False, CFL = 0.6, **kwargs):
        """
        Initialisation d'une instance de Donnee2D, permettant à la résolution du problème 2D de proagation en milieu homogène ou modulée en temps
        """
        self.c: float = c                                              # célérité en m/s
        self.rho: float = rho                                          # masse volumique en g/m^3
        self.kappa : float = kappa                                             #module d'Young en Pa
        self.f: float = f                                              # frequence max observable
        self.xc: tuple = xc                                            # couple position x min, max
        self.yc: tuple = yc                                            # couple position y min, max
        self.tc: tuple = tc                                            # couple temps init, final
        self.Mx: int = Mx                                              # discretisation spatiale selon x
        self.My: int = My                                              # discretisation spatiale selon y
        if self.Mx != self.My: print("Mx != My")                       # alerte d'asymétrie
        self.dx: float = self.xc[1] / self.Mx                          # infinitesimal spatial en x
        self.dy: float = self.xc[1] / self.My                          # infinitesimal spatial en y
        self.CFL = CFL
        self.dt: float = self.CFL * max(self.dx,self.dy) / self.c      # infinitesimal temporel
        self.opt: bool = opt                                           # True si perturb sur vitesse, False si perturb sur pression

        if "label" in kwargs.keys(): self.label = kwargs["label"]      # label
        else: self.label = ""

        self.N: int = int(self.tc[1] / self.dt)                        # discretisation temporelle
        self.x = np.linspace(-self.xc[1]/2, self.xc[1]/2, self.Mx)          # axe x
        self.y = np.linspace(-self.yc[1]/2, self.yc[1]/2, self.My)          # axe y
        self.t = np.linspace(self.tc[0], self.tc[1], self.N)           # axe temporel

        if "U" in kwargs.keys(): self.U: np.ndarray() = kwargs["U"]    # vecteur [v, w, p] solution
        else: self.U: np.ndarray() = np.zeros((self.N, self.Mx, self.My, 2)) #U(t, x, y)

        self.fmax: float = self.f * 10

        if "S" in kwargs.keys(): self.S = kwargs["S"]                  # donnee point source
        else: self.S = pt_source_2D

        if "ps" in kwargs.keys(): self.ps = kwargs["ps"]               # postion point source
        else: self.ps : [(int,int)] = [(self.Mx//2, self.My//2)]

        if "E" in kwargs.keys():self.E: np.ndarray() = kwargs["E"]     # energie
        else: self.E: np.ndarray() = np.zeros(self.N)

        if "opt" in kwargs.keys(): self.opt = kwargs["opt"]                #True si perturb sur vitesse, False si perturb sur pression
        else: self.opt : bool = True

        if "eps_r" in kwargs.keys(): self.eps_r = kwargs["eps_r"]                #Modulation temporelle
        else: self.eps_p = 0
        if "eps_kappa" in kwargs.keys(): self.eps_kappa = kwargs["eps_kappa"]                #Modulation temporelle
        else: self.eps_kappa = 0
        if "omega" in kwargs.keys(): self.omega = kwargs["omega"]
        else: self.omega = 0

        if "rho_mt" in kwargs.keys(): self.rho_mt = kwargs["rho_mt"]
        else: self.rho_mt = rho_sinus
        if "kappa_mt" in kwargs.keys(): self.kappa_mt = kwargs["kappa_mt"]
        else: self.kappa_mt = kappa_sinus

        if "alpha" in kwargs.keys(): self.alpha = kwargs["alpha"]
        else: self.alpha = 0.5

        if "param" in kwargs.keys(): self.param = kwargs["param"]
        else: self.param = 0.5

    def __str__(self):
        f = self.omega / (2 * np.pi)
        crit = self.c / ((f * 4 + self.f) * self.dx)
        mess = ""
        if crit <= 25:
            mess = " -> ATTENTION ! TROP PETIT, AUGEMENTER M"
        crit = str(crit) + mess
        return "\nDonnee2D(" + self.label + ", c :" + str(self.c) + ", kappa : " + str(self.kappa) + ", rho : " + str(
                self.rho) + ",\n" + "N : " + str(self.N) + ", Mx : " + str(self.Mx) + ", My : " + str(self.My) + ", dx : " + str(
                self.dx) +  ", dy : " + str(self.dy) + ', dt : ' + str(self.dt) + ")\nlambda/dx = " + crit

    def __eq__(self, other):
        return (self.c == other.c)*(self.rho == other.rho)*(self.Mx == other.Mx)*(self.My == other.My)*(self.CFL == other.CFL)*(self.xc == other.xc)*(self.yc == other.yc)*(self.tc == other.tc)*(self.f == other.f)*(self.S == other.S)*(self.eps_r == other.eps_r)*(self.eps_kappa == other.eps_kappa)*(self.omega == other.omega)*(self.rho_mt == other.rho_mt)*(self.kappa_mt == other.kappa_mt)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __sub__(self, other):
        ecart = self.U - other.U
        return np.array([[np.max(ecart[...,0]), np.min(ecart[...,0]), np.average(ecart[...,0]), np.std(ecart[...,0])],
                         [np.max(ecart[...,1]), np.min(ecart[...,1]), np.average(ecart[...,1]), np.std(ecart[...,1])],
                         [np.max(ecart[...,2]), np.min(ecart[...,2]), np.average(ecart[...,2]), np.std(ecart[...,2])]])

    def copy(self, instance : Donnee2D):                               #constructeur de recopie
        self.c : float = instance.c
        self.rho: float = instance.rho
        self.f: float = instance.f
        self.xc: tuple = instance.xc
        self.yc: tuple = instance.yc
        self.tc: tuple = instance.tc
        self.Mx: int = instance.Mx
        self.My: int = instance.My
        self.dx: float = instance.dx
        self.dy: float = instance.dy
        self.CFL: float = instance.CFL
        self.dt: float = instance.dt
        self.N : int = instance.N
        self.U : np.ndarray = instance.U
        self.fmax: float = self.f * 10
        self.opt: bool = instance.opt
        self.label: str = instance.label
        self.S = instance.S
        self.ps : list[tuple] = instance.ps
        self.E = instance.E

    def projection(self, coupe):
        """
        Réalise la projection d'une variable de Donnee2D en une variable de Donnee1D
        :param coupe: (int,int), correspond à la positon de la coupe. Attention, une des deux coordonnées doit forcément être nulle !!
        :return: Donnee1D
        """
        coupe = (int(coupe[0]), int(coupe[1]))
        pos = "-(x:" + str(coupe[0]) + ",y:" + str(coupe[1]) + ")"
        try:
            retour = Donnee1D(c=self.c, rho= self.rho, kappa= self.kappa, f=self.f, tc=self.tc, eps_r = self.eps_r, alpha = self.alpha, eps_E = self.eps_kappa, omega = self.omega,
                              M=self.Mx, CFL = self.CFL, xc = self.xc, rho_mt = self.rho_mt, E_mt = self.kappa_mt,
                              fmax=self.f * 10, opt=self.opt, label=self.label + pos, S=self.S, t = self.t)
        except:
            retour = Donnee1D(c=self.c, rho=self.rho, kappa=self.kappa, f=self.f, tc=self.tc,
                              M=self.Mx, CFL = self.CFL, xc = self.xc,
                              fmax=self.f * 10, opt=self.opt, label=self.label + pos, S=self.S, t = self.t)
        if coupe[0] != 0:  # coupe selon l'axe y
            retour.U = self.U[:, coupe[0], :, 1:]
            retour.xc = self.yc
            retour.x = self.y + self.yc[1]/2
            if type(self.rho) == tuple:
                ec = 0.5 * (self.rho[0] * retour.U[..., 0] ** 2)
                ep = 0.5 * retour.U[..., 1] ** 2 / (2*self.kappa)
                retour.E = np.sum((ec + ep)*self.dx, axis = 1)
            else:
                retour.calcul_energie()
        if coupe[1] != 0:  # coupe selon l'axe x
            retour.U = self.U[:, :, coupe[1], ::2]
            retour.xc = self.xc
            retour.x = self.x + self.xc[1]/2
            if type(self.rho) == tuple:
                ec = 0.5 * (self.rho[1] * retour.U[..., 0] ** 2)
                ep = 0.5 * retour.U[..., 1] ** 2 / (2*self.kappa)
                retour.E = np.sum((ec + ep)*self.dx, axis = 1)
            else:
                retour.calcul_energie()
        return retour

    def CFL_maj(self, **kwargs):
        """
        Permet de corriger la CFL dans un milieu modulé en temps
        :return: None
        """
        if "c" in kwargs.keys():
            self.c = kwargs["c"]
        else:
            rho, kappa = [], []
            for t in self.t:
                rho.append(self.rho * self.rho_mt(self)(t)[0])
                kappa.append(self.kappa * self.kappa_mt(self)(t)[0])
            rho, kappa = np.array(rho), np.array(kappa)
            self.c: float = np.max(np.sqrt(kappa/rho))

        self.dt : float = self.CFL * self.dx / self.c
        self.N: int = int(self.tc[1] / self.dt)
        self.t = np.linspace(self.tc[0], self.tc[1], self.N)

    def CFL_aniso(self, c):
        """
        Permet de corriger la CFL dans un milieu anisotropique non modulé en temps
        :return: None
        """
        self.c = c
        self.dt: float = self.CFL * self.dx / self.c
        self.N: int = int(self.tc[1] / self.dt)
        self.t = np.linspace(self.tc[0], self.tc[1], self.N)

    def calcul_energie(self):
        if type(self.rho) == float:
            ec = 0.5 * self.rho * (self.U[..., 0] ** 2 + self.U[..., 1] ** 2)
            ep = self.U[..., 2] ** 2 / (2*self.rho * self.c ** 2)
            self.E = np.sum((ec + ep)*self.dx*self.dy, axis=(1, 2))
        elif type(self.rho) == tuple:
            (r0, r1) = self.rho
            r00 = r0 * self.alpha + r1 * (1 - self.alpha)
            r11 = r0 * r1 / (self.alpha * r1 + r0 * (1 - self.alpha))
            e = 1 / (self.alpha / self.kappa[0] + (1 - self.alpha) / self.kappa[1])
            ec = 0.5 * (r00 * self.U[..., 0] ** 2 + r11 * self.U[..., 1] ** 2)
            ep = 0.5 * self.U[..., 2] ** 2 / (2*e)
            self.E = np.sum((ec + ep)*self.dx*self.dy, axis=(1, 2))

    def aniso(self):
        (r0, r1) = self.rho
        r00 = r0 * self.alpha + r1 * (1 - self.alpha)
        r11 = r0 * r1 / (self.alpha * r1 + r0 * (1 - self.alpha))
        kappa = 1 / (self.alpha / self.kappa[0] + (1 - self.alpha) / self.kappa[1])
        self.kappa = kappa
        self.rho = (r00, r11)