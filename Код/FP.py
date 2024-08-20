from galerkin import Solver_2D
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class Solver_FP:
    def __init__(self, f, xmax, m, s, n, acc=0, p=0, path=''):
        self.xmax = xmax
        self.arrX = [x for x in range(xmax + 1)]
        self.arrK = [np.random.normal(m, s) for _ in range(n)]

        print(f"ksi = {self.arrK}", end='\n\n')

        self.m = np.mean(self.arrK)
        self.s = np.sqrt(np.var(self.arrK))

        self.f = f
        self.Dx = np.arange(0, xmax, 0.02 * xmax)
        
        solver = Solver_2D(f, self.arrX, self.arrK, acc=acc, p=p)
        self.galerkin = solver.galerkin()

        def wP(ksi):
            N = 1 / integrate.quad(lambda y: f(y, ksi), 0, np.inf)[0]
            return lambda x: N * f(x, ksi)
        
        self.wP = wP
        
        def wG(ksi):
            N = 1 / integrate.quad(lambda y: self.galerkin(y, ksi), 0, self.xmax)[0]
            return lambda x: N * self.galerkin(x, ksi)
        
        self.wG = wG

        def wF(ksi):
            P = self.wP(ksi)
            return lambda x: integrate.quad(lambda y: P(y), 0, x)[0]
        
        self.wF = wF

        def wGF(ksi):
            G = self.wG(ksi)
            return lambda x: integrate.quad(lambda y: G(y), 0, x)[0]
        
        self.wGF = wGF

        N = 1 / integrate.quad(lambda y: f(y, self.m), 0, np.inf)[0]
        self.mP = lambda x: N * f(x, self.m)
        print(f"N = {N}", end='\n\n')

        gN = 1 / integrate.quad(lambda y: self.galerkin(y, self.m), 0, self.xmax)[0]
        self.mG = lambda x: gN * self.galerkin(x, self.m)
        print(f"gN = {gN}", end='\n\n')

        self.mF = lambda x: integrate.quad(lambda y: self.mP(y), 0, x)[0]
        self.mGF = lambda x: integrate.quad(lambda y: self.mG(y), 0, x)[0]

        self.path = path

    
    def check(self, *, acc):
        for x in self.arrX:
            if self.mG(x) < 0 or self.mGF(x) < 0 or self.mGF(x) > 1:
                return False
        
        return sum([(self.mGF(x) - self.mF(x)) ** 2 for x in self.Dx]) / len(self.Dx) < acc
    
    def draw_middle_section(self):

        #err_hermite = sum([(self.G(x, ksi) - self.P(x, ksi)) ** 2 for ksi in self.arrK for x in self.arrX]) / (len(self.arrX) * len(self.arrK))
        #print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита равна {err_hermite}", end='\n\n')

        print(f"mksi = {self.m}", end='\n\n')
        print(f"sksi = {self.s}", end='\n\n')

        '''Dk = np.arange(self.m - 3 * self.s, self.m + 3 * self.s, 6 * self.s / (len(self.arrK) - 1))
        xgrid, kgrid = np.meshgrid(Dx, Dk)

        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        
        axes.plot_surface(xgrid, kgrid, self.G(xgrid, kgrid), color="red")
        axes.plot_surface(xgrid, kgrid, self.P(xgrid, kgrid), color="navy")

        axes.set_xlabel("X")
        axes.set_ylabel("ksi")
        axes.set_zlabel("Z")

        plt.show()'''

        plt.grid(True)
        
        plt.plot(self.Dx, [self.mG(x) for x in self.Dx], color="red", label="Аппроксимирующая функция")
        plt.plot(self.Dx, [self.mP(x) for x in self.Dx], color="navy", label="Исходная функция")

        err2 = sum([(self.mG(x) - self.mP(x)) ** 2 for x in self.arrX]) / len(self.arrX)
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита (2) равна {err2}", end='\n\n')

        plt.legend(loc='best')
        plt.savefig(self.path + 'density_middle_section.jpg', dpi=500)
        plt.close()

        print(f"Интеграл от исходной функции равен {self.mF(self.xmax)}", end='\n\n')
        print(f"Интеграл от аппроксимации равен {self.mGF(self.xmax)}", end='\n\n')

        err3 = sum([(self.mGF(x) - self.mF(x)) ** 2 for x in self.arrX]) / len(self.arrX)
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита (3) равна {err3}", end='\n\n')

        plt.grid(True)
        
        plt.plot(self.Dx, [self.mF(x) for x in self.Dx], color="navy", label="Исходная функция")
        plt.plot(self.Dx, [self.mGF(x) for x in self.Dx], color="red", label="Аппроксимирующая функция")

        plt.legend(loc='best')
        plt.savefig(self.path + 'distribution_middle_section.jpg', dpi=500)
        plt.close()
    
    def draw_model(self):
        arrK = [ksi for ksi in self.arrK if np.abs(ksi - self.m) < self.s]

        plt.grid(True)
        
        for ksi in arrK:
            P = self.wP(ksi)
            plt.plot(self.Dx, [P(x) for x in self.Dx], alpha=0.2, color="navy")  

        plt.savefig(self.path + 'density_model.jpg', dpi=500)
        plt.close()
        
        plt.grid(True)
        
        for ksi in arrK:
            F = self.wF(ksi)
            plt.plot(self.Dx, [F(x) for x in self.Dx], alpha=0.2, color="navy")
        
        plt.savefig(self.path + 'distribution_model.jpg', dpi=500)
        plt.close()
    
    def draw_PC(self):
        arrK = [ksi for ksi in self.arrK if np.abs(ksi - self.m) < self.s]

        plt.grid(True)
        
        for ksi in arrK:
            G = self.wG(ksi)
            plt.plot(self.Dx, [G(x) for x in self.Dx], alpha=0.2, color="red") 
        
        plt.savefig(self.path + 'density_PC.jpg', dpi=500)
        plt.close()
        
        plt.grid(True)
        
        for ksi in arrK:
            GF = self.wGF(ksi)
            plt.plot(self.Dx, [GF(x) for x in self.Dx], alpha=0.2, color="red")
        
        plt.savefig(self.path + 'distribution_PC.jpg', dpi=500)
        plt.close()
    
    def draw_combined(self):
        arrK = [ksi for ksi in self.arrK if np.abs(ksi - self.m) < self.s]

        plt.grid(True)
        
        for ksi in arrK:
            P = self.wP(ksi)
            G = self.wG(ksi)
            plt.plot(self.Dx, [P(x) for x in self.Dx], alpha=0.2, color="navy")
            plt.plot(self.Dx, [G(x) for x in self.Dx], alpha=0.2, color="red")
        
        plt.savefig(self.path + 'density_combined.jpg', dpi=500)
        plt.close()
        
        plt.grid(True)
        
        for ksi in arrK:
            F = self.wF(ksi)
            GF = self.wGF(ksi)
            plt.plot(self.Dx, [F(x) for x in self.Dx], alpha=0.2, color="navy")
            plt.plot(self.Dx, [GF(x) for x in self.Dx], alpha=0.2, color="red")
        
        plt.savefig(self.path + 'distribution_combined.jpg', dpi=500)
        plt.close()
       

class Solver_BFP(Solver_FP):
    def __init__(self, X, k4, *, xmax, m, s, n, acc=0, p=0):
        G = lambda ksi: 2 * X[0](ksi) * (X[0](ksi) + X[1](ksi)) * (X[0](ksi) + X[2](ksi)) / ((X[0](ksi) - X[1](ksi)) * (X[0](ksi) - X[2](ksi)))
        
        H = lambda ksi: 2 * X[1](ksi) * (X[0](ksi) + X[1](ksi)) * (X[1](ksi) + X[2](ksi)) / ((X[0](ksi) - X[1](ksi)) * (X[1](ksi) - X[2](ksi)))
        
        M = lambda ksi: 2 * X[2](ksi) * (X[0](ksi) + X[2](ksi)) * (X[1](ksi) + X[2](ksi)) / ((X[0](ksi) - X[2](ksi)) * (X[2](ksi) - X[1](ksi)))
        
        print(f"G(1) * m = {G(1) * m}", end='\n\n')
        print(f"H(1) * m = {H(1) * m}", end='\n\n')
        print(f"M(1) * m = {M(1)* m}", end='\n\n')

        B = lambda x, ksi: k4 * (x + X[0](ksi)) * (x + X[1](ksi)) * (x + X[2](ksi))

        f = lambda x, ksi: np.exp(2 * (G(ksi) * np.log(1 + x / X[0](ksi)) - H(ksi) * np.log(1 + x / X[1](ksi)) - M(ksi) * np.log(1 + x / X[2](ksi)) - x)) / B(x, ksi)

        super().__init__(f, xmax, m, s, n, acc, p, path='results/BFP/')      
    
class Solver_MFP(Solver_FP):
    def __init__(self, K, *, xmax, m, s, n, acc=0, p=0):
        def f(x, ksi):
            A = lambda y: K[0](ksi) - K[1](ksi) * y + K[2](ksi) * y * y - K[3](ksi) * y ** 3
            B = lambda y: K[0](ksi) + K[1](ksi) * y + K[2](ksi) * y * y + K[3](ksi) * y ** 3
            g = lambda y: 2 * integrate.quad(lambda u: A(u) / B(u), 0, y)[0]

            return np.exp(g(x)) / B(x)
        
        super().__init__(f, xmax, m, s, n, acc, p, path='results/MFP/')
        
        self.M = lambda ksi: integrate.quad(lambda x: x * self.wP(ksi)(x), 0, np.inf)
        self.GM = lambda ksi: integrate.quad(lambda x: x * self.wG(ksi)(x), 0, self.xmax)
    
    def draw_expectation(self):
        plt.grid(True)

        Dksi = np.arange(self.m - 2 * self.s, self.m + 2 * self.s, 0.1 * 4 * self.s)

        plt.plot(Dksi, [self.M(ksi) for ksi in Dksi], color="navy", label="Математическое ожидание исходной функции")
        plt.plot(Dksi, [self.GM(ksi) for ksi in Dksi], color="red", label="Математическое ожидание аппроксимирующей функции")
        plt.legend(loc='best')

        plt.savefig(self.path + 'expectation.jpg', dpi=500)
        plt.close()
    

class Solver_BFP2(Solver_FP):
    def __init__(self, K, *, xmax, m, s, n, acc=0, p=0):
        def f(x, ksi):
            A = lambda y: K[0](ksi) - K[1](ksi) * y + K[2](ksi) * y * y - K[3](ksi) * y ** 3
            B = lambda y: K[0](ksi) + K[1](ksi) * y + K[2](ksi) * y * y + K[3](ksi) * y ** 3
            g = lambda y: 2 * integrate.quad(lambda u: A(u) / B(u), 0, y)[0]

            return np.exp(g(x)) / B(x)
        
        super().__init__(f, xmax, m, s, n, acc, p, path='results/BFP2/')


if __name__ == "__main__":
    seed = 100009
    np.random.seed(seed)

    '''def x1(ksi):
        return ksi
    
    def x2(ksi):
        return 2 * ksi
    
    def x3(ksi):
        return 4 * ksi

    X = [x1, x2, x3]

    k4 = 0.001

    fp = Solver_BFP(X, k4, xmax=50, m=10, s=1, n=10, p=18)'''

    def k1(ksi):
        return 0.1 * (1 + 4 * ksi)
    
    def k2(ksi):
        return 0.1 * (1 + 2 * ksi)
    
    def k3(ksi):
        return 0.1 * (1 + ksi)
    
    def k4(ksi):
        return 0.01 * ksi
    
    K = [k1, k2, k3, k4]

    fp = Solver_MFP(K, xmax=30, m=1, s=0.1, n=20, p=18)

    fp.draw_expectation()

    '''def k1(ksi):
        return 8 * ksi
    
    def k2(ksi):
        return 1.4 * ksi
    
    def k3(ksi):
        return 0.07 * ksi
    
    def k4(ksi):
        return 0.001
    
    K = [k1, k2, k3, k4]

    fp = Solver_BFP2(K, xmax=50, m=1, s=0.1, n=20, p=18)'''

    '''while not fp.check(acc=0.01):
        print(seed, end='\n\n')
        seed += 1
        np.random.seed(seed)
        fp = Solver_MFP(K, xmax=30, m=1, s=0.1, n=20, p=18)

    print(f"seed = {seed}", end='\n\n')'''

    fp.draw_middle_section()
    fp.draw_model()
    fp.draw_PC()
    fp.draw_combined()

    print("Завершено!", end='\n\n')
