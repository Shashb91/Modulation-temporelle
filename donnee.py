from source import*
from math import sqrt

def norme(a,b):
    return sqrt(a**2 + b**2)/sqrt(2)

class Donnee1D:
    def __init__(self, c = 2800, rho = 1200, f = 20, xc = (0,400), tc = (0, 0.125), M = 400, CFL = 0.95, **kwargs):
        self.c : float = c                                                 #célérité en m/s
        self.rho : float = rho                                             #masse volumique en g/m^3
        self.f : float = f                                                 #frequence max observable
        self.xc : tuple = xc                                               #couple position min, max
        self.tc : tuple = tc                                               #couple temps init, final
        self.M : int = M                                                   #discretisation spatiale
        self.dx : float = self.xc[1]/self.M                                #infinitesimal spatial
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
        else: self.S = pt_source_vitesse

        if "xs" in kwargs.keys(): self.xs : int = kwargs["xs"]             #postion point source
        else: self.xs : int = self.M // 2

        if "E" in kwargs.keys(): self.E : np.ndarray() = kwargs["E"]       #energie
        else: self.E : np.ndarray() = np.zeros(self.N)

        if "opt" in kwargs.keys(): self.opt = kwargs["opt"]                #True si perturb sur vitesse, False si perturb sur pression
        else: self.opt : bool = True

class Donnee2D:
    def __init__(self, c=2800, rho=1200, f=20, xc=(0, 1000), yc = (0, 1000), tc=(0, 0.25), Mx = 300, My = 300, opt = False, CFL = 0.95, **kwargs):
        self.c: float = c                                              # célérité en m/s
        self.rho: float = rho                                          # masse volumique en g/m^3
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
        self.dt: float = self.CFL * max(self.dx,self.dy) / (self.c)    # infinitesimal temporel
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
        else:
            if opt : self.S = pt_source_vitesse
            else: self.S = pt_source_pression

        if "xs" in kwargs.keys(): self.xs: int = kwargs["xs"]          # postion point source
        else: self.xs: int = self.Mx // 2
        if "ys" in kwargs.keys(): self.ys: int = kwargs["ys"]
        else: self.ys: int = self.My // 2

        if "E" in kwargs.keys():self.E: np.ndarray() = kwargs["E"]     # energie
        else: self.E: np.ndarray() = np.zeros(self.N)