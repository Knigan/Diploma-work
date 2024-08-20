from galerkin import Solver_2D
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class Solver_FP:
    def __init__(self, f, xmax, m, s, n, acc=0, p=0, path=''):
        self.xmax = xmax
        self.arrX = [x for x in range(xmax + 1)]
        self.arrK = [np.random.normal(m, s) for _ in range(n)]
        self.arrK.sort()

        print(f"ksi = {self.arrK}", end='\n\n')

        self.m = np.mean(self.arrK)
        self.s = np.sqrt(np.var(self.arrK))

        N = 1 / integrate.quad(lambda x: f(x, np.mean(self.arrK)), 0, np.inf)[0]
        print(f"N = {N}", end='\n\n')
        

        self.P = lambda x, ksi: N * f(x, ksi)
        solver = Solver_2D(self.P, self.arrX, self.arrK, acc=acc, p=p)
        self.galerkin = solver.galerkin()

        self.F = lambda x, ksi: integrate.quad(lambda y: self.P(y, ksi), 0, x)[0]
        self.GF = lambda x, ksi: integrate.quad(lambda y: self.galerkin(y, ksi), 0, x)[0]

        self.path = path
    
    def check(self, *, acc=0.01):
        for x in self.arrX:
            if self.galerkin(x, self.m) < 0:
                return False
            
        for x in self.arrX:
            if self.GF(x, self.m) < 0 or self.GF(x, self.m) > 1:
                return False

        I1 = integrate.quad(lambda x: self.P(x, self.m), 0, self.xmax)[0]
        I2 = integrate.quad(lambda x: self.galerkin(x, self.m), 0, self.xmax)[0]

        return np.abs(I2 - I1) < acc
    
    def draw(self):
        err_hermite = sum([(self.galerkin(x, ksi) - self.P(x, ksi)) ** 2 for ksi in self.arrK for x in self.arrX]) / (len(self.arrX) * len(self.arrK))
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита равна {err_hermite}", end='\n\n')

        Dx = np.arange(self.arrX[0], self.arrX[-1], (self.arrX[-1] - self.arrX[0]) / (len(self.arrX) - 1))

        print(f"mksi = {self.m}", end='\n\n')
        print(f"sksi = {self.s}", end='\n\n')

        Dk = np.arange(self.m - 3 * self.s, self.m + 3 * self.s, 6 * self.s / (len(self.arrK) - 1))
        xgrid, kgrid = np.meshgrid(Dx, Dk)

        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        
        axes.plot_surface(xgrid, kgrid, self.galerkin(xgrid, kgrid), color="red")
        #axes.plot_surface(xgrid, kgrid, self.P(xgrid, kgrid), color="navy")

        axes.set_xlabel("X")
        axes.set_ylabel("ksi")
        axes.set_zlabel("Z")

        plt.show()

        plt.grid(True)

        plt.plot(Dx, self.galerkin(Dx, self.m), color="red", label="Аппроксимирующая функция")
        plt.plot(Dx, [self.P(x, self.m) for x in Dx], color="navy", label="Исходная функция")

        err2 = sum([(self.galerkin(x, self.m) - self.P(x, self.m)) ** 2 for x in self.arrX]) / len(self.arrX)
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита (2) равна {err2}", end='\n\n')

        plt.legend(loc='best')
        plt.savefig(self.path + 'density_middle_section.jpg', dpi=500)
        plt.close()

        plt.grid(True)

        I1 = integrate.quad(lambda x: self.P(x, self.m), 0, self.xmax)[0]
        I2 = integrate.quad(lambda x: self.galerkin(x, self.m), 0, self.xmax)[0]
        print(f"Интеграл от исходной функции равен {I1}", end='\n\n')
        print(f"Интеграл от аппроксимации равен {I2}", end='\n\n')

        err3 = sum([(self.GF(x, self.m) - self.F(x, self.m)) ** 2 for x in self.arrX]) / len(self.arrX)
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита (3) равна {err3}", end='\n\n')

        plt.plot(Dx, [self.F(x, self.m) for x in Dx], color="navy", label="Исходная функция")
        plt.plot(Dx, [self.GF(x, self.m) for x in Dx], color="red", label="Аппроксимирующая функция")

        plt.legend(loc='best')
        plt.savefig(self.path + 'distribution_middle_section.jpg', dpi=500)
        plt.close()
    
    def draw_model(self):
        plt.grid(True)
        Dx = np.arange(self.arrX[0], self.arrX[-1], (self.arrX[-1] - self.arrX[0]) / (len(self.arrX) - 1))
        arrK = [ksi for ksi in self.arrK if np.abs(ksi - self.m) < self.s]

        for ksi in arrK:
            plt.plot(Dx, [self.P(x, ksi) for x in Dx], alpha=0.1, color="navy")

        plt.savefig(self.path + 'density_model.jpg', dpi=500)
        plt.close()

        plt.grid(True)

        for ksi in arrK:
            plt.plot(Dx, [self.F(x, ksi) for x in Dx], alpha=0.1, color="navy")
        
        plt.plot(Dx, [self.F(x, self.m) for x in Dx], color="navy")
        
        plt.savefig(self.path + 'distribution_model.jpg', dpi=500)
        plt.close()
    
    def draw_PC(self):
        plt.grid(True)
        Dx = np.arange(self.arrX[0], self.arrX[-1], (self.arrX[-1] - self.arrX[0]) / (len(self.arrX) - 1))
        arrK = [ksi for ksi in self.arrK if np.abs(ksi - self.m) < self.s]

        for ksi in arrK:
            plt.plot(Dx, [self.galerkin(x, ksi) for x in Dx], alpha=0.1, color="red")

        plt.savefig(self.path + 'density_PC.jpg', dpi=500)
        plt.close()

        plt.grid(True)
        
        for ksi in arrK:
            plt.plot(Dx, [self.GF(x, ksi) for x in Dx], alpha=0.1, color="red")
        
        plt.plot(Dx, [self.GF(x, self.m) for x in Dx], color="red")
        
        plt.savefig(self.path + 'distribution_PC.jpg', dpi=500)
        plt.close()
    
    def draw_combined(self):
        plt.grid(True)
        Dx = np.arange(self.arrX[0], self.arrX[-1], (self.arrX[-1] - self.arrX[0]) / (len(self.arrX) - 1))
        arrK = [ksi for ksi in self.arrK if np.abs(ksi - self.m) < self.s]

        for ksi in arrK:
            plt.plot(Dx, [self.P(x, ksi) for x in Dx], alpha=0.1, color="navy")
            plt.plot(Dx, [self.galerkin(x, ksi) for x in Dx], alpha=0.1, color="red")        
        
        plt.savefig(self.path + 'density_combined.jpg', dpi=500)
        plt.close()
        
        plt.grid(True)

        for ksi in arrK:
            plt.plot(Dx, [self.F(x, ksi) for x in Dx], alpha=0.1, color="navy")
            plt.plot(Dx, [self.GF(x, ksi) for x in Dx], alpha=0.1, color="red")
        
        plt.plot(Dx, [self.F(x, self.m) for x in Dx], color="navy")
        plt.plot(Dx, [self.GF(x, self.m) for x in Dx], color="red")

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

        def B(x, ksi):
            return k4 * (x + X[0](ksi)) * (x + X[1](ksi)) * (x + X[2](ksi))
        
        def f(x, ksi):
            return np.exp(2 * (G(ksi) * np.log(1 + x / X[0](ksi)) - H(ksi) * np.log(1 + x / X[1](ksi)) - M(ksi) * np.log(1 + x / X[2](ksi)) - x)) / B(x, ksi)

        super().__init__(f, xmax, m, s, n, acc, p, path='results/BFP/')      
    
class Solver_MFP(Solver_FP):
    def __init__(self, K, *, xmax, m, s, n, acc=0, p=0):
        def f(x, ksi):
            A = lambda y: K[0](ksi) - K[1](ksi) * y + K[2](ksi) * y * y - K[3](ksi) * y ** 3
            B = lambda y: K[0](ksi) + K[1](ksi) * y + K[2](ksi) * y * y + K[3](ksi) * y ** 3
            g = lambda y: 2 * integrate.quad(lambda u: A(u) / B(u), 0, y)[0]

            return np.exp(g(x)) / B(x)
        
        super().__init__(f, xmax, m, s, n, acc, p, path='results/MFP/')


if __name__ == "__main__":
    seed = 118848
    np.random.seed(seed)

    def x1(ksi):
        return ksi
    
    def x2(ksi):
        return 2 * ksi
    
    def x3(ksi):
        return 4 * ksi

    X = [x1, x2, x3]

    k4 = 0.001

    fp = Solver_BFP(X, k4, xmax=50, m=10, s=1, n=10, p=18)

    '''def k1(ksi):
        return 0.1 * (1 + 4 * ksi)
    
    def k2(ksi):
        return 0.1 * (1 + 2 * ksi)
    
    def k3(ksi):
        return 0.1 * (1 + ksi)
    
    def k4(ksi):
        return 0.01 * ksi
    
    K = [k1, k2, k3, k4]

    fp = Solver_MFP(K, xmax=30, m=1, s=0.01, n=20, p=18)'''

    '''while not fp.check(acc=0.01):
        print(seed, end='\n\n')
        seed += 1
        np.random.seed(seed)
        fp = Solver_BFP(X, k4, xmax=50, m=10, s=1, n=10, p=18)'''

    print(f"seed = {seed}", end='\n\n')

    fp.draw()
    fp.draw_model()
    fp.draw_PC()
    fp.draw_combined()

    print("Выполнено!")