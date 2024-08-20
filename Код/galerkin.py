import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class Solver:
    @staticmethod
    def He(n, x):
        def factorial(num):
            res = 1
            for k in range(2, num + 1):
                res *= k   
            return res
        
        S = 0
        for j in range(int(0.5 * n) + 1):
            S += (-0.5) ** j * np.sqrt(factorial(n)) / (factorial(j) * factorial(n - 2 * j)) * x ** (n - 2 * j)    
        return S
    
    def draw(self):
        pass

class Solver_1D(Solver):
    
    def __init__(self, f, X, *, acc=0, p=0, silent=False):
        self.f = f
        self.X = X
        self.X.sort()

        self.m = np.mean(self.X)
        self.s = np.sqrt(np.var(self.X))
        self.acc = acc

        self.p = p

        self.silent = silent


    def galerkin(self):
        def a(j):
            def h(k):
                return self.X[k + 1] - self.X[k]
            
            def I(k):
                return self.f(self.X[k]) * Solver.He(j, (self.X[k] - self.m) / self.s) * np.exp(-(self.X[k] - self.m) ** 2 / (2 * self.s * self.s))
            
            S = 0
            for k in range(len(self.X) - 1):
                S += 0.5 * h(k) * (I(k) + I(k + 1))
            
            return S / (np.sqrt(2 * np.pi) * self.s)

        
        A = []
        j = 0
        if self.p > 0:
            while j <= self.p:
                A.append(a(j))
                j += 1
        elif self.acc > 0:
            k = 0
            while k < 6:
                x = a(j)
                A.append(x)
                if np.abs(x) < self.acc:
                    k += 1
                else:
                    k = 0
                j += 1
        else:
            raise ValueError("p или acc должны быть положительным числом")
        
        if not self.silent:
            print("Коэффициенты в разложении полиномиального хаоса: ", A, end='\n\n')
            print(f"Разложение производится до {len(A) - 1} порядка", end='\n\n')

        def Fest(x):
            S = 0
            for j in range(len(A)):
                S += A[j] * Solver.He(j, (x - self.m) / self.s)
            
            return S
        
        return Fest

    def least_squares(self):
        if self.p == 0:
            flag = False
            N = 6
            while not flag:
                M = [[sum([self.X[i] ** (k + j) for i in range(len(self.X))]) for k in range(N + 1)] for j in range(N + 1)]
                V = [sum([self.f(self.X[i]) * self.X[i] ** j for i in range(len(self.X))]) for j in range(N + 1)]

                A = np.linalg.solve(M, V)
                if max(map(abs, A[-6:])) < self.acc:
                    flag = True
                else:
                    flag = False
                
                N += 1
        else:
            M = [[sum([self.X[i] ** (k + j) for i in range(len(self.X))]) for k in range(self.p + 1)] for j in range(self.p + 1)]
            V = [sum([self.f(self.X[i]) * self.X[i] ** j for i in range(len(self.X))]) for j in range(self.p + 1)]    
            A = np.linalg.solve(M, V)
        
        if not self.silent:
            print("Коэффициенты ряда: ", A, end='\n\n')
            print(f"Разложение производится до {len(A) - 1} порядка", end='\n\n')
        
        def Fest(x):
            S = 0
            for k in range(len(A)):
                S += A[k] * x ** k
            return S
        
        return Fest

    def compare_errors(self):
        D = np.linspace(self.m - 3 * self.s, self.m + 3 * self.s, len(self.X))

        g = self.galerkin()
        ls = self.least_squares()

        '''plt.plot(D, [g(x) for x in D], color="red")
        plt.plot(D, [ls(x) for x in D], color="green")
        plt.plot(D, [self.f(x) for x in D], color="navy")

        err_galerkin = sum([(self.f(x) - g(x)) ** 2 for x in D]) / len(D)
        err_ls = sum([(self.f(x) - ls(x)) ** 2 for x in D]) / len(D)'''

        Ih = lambda x: integrate.quad(g, 0, x)[0]
        Ik = lambda x: integrate.quad(ls, 0, x)[0]
        If = lambda x: integrate.quad(self.f, 0, x)[0]

        err_galerkin = sum([(If(x) - Ih(x)) ** 2 for x in D]) / len(D)
        err_ls = sum([(If(x) - Ik(x)) ** 2 for x in D]) / len(D)

        if not self.silent:
            print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита равна {err_galerkin}", end='\n\n')
            print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Колмогорова-Габора равна {err_ls}", end='\n\n')
        

        return (err_galerkin, err_ls)

    def compare(self):
        plt.grid(True)
        D = np.linspace(self.m - 3 * self.s, self.m + 3 * self.s, len(self.X))

        g = self.galerkin()
        ls = self.least_squares()

        plt.plot(D, [g(x) for x in D], color="red")
        plt.plot(D, [ls(x) for x in D], color="green")
        plt.plot(D, [self.f(x) for x in D], color="navy")

        err_galerkin = sum([(self.f(x) - g(x)) ** 2 for x in D]) / len(D)
        err_ls = sum([(self.f(x) - ls(x)) ** 2 for x in D]) / len(D)
        
        '''Ih = lambda x: integrate.quad(g, 0, x)[0]
        Ik = lambda x: integrate.quad(ls, 0, x)[0]
        If = lambda x: integrate.quad(self.f, 0, x)[0]

        plt.plot(D, [Ih(x) for x in D], color="red")
        plt.plot(D, [Ik(x) for x in D], color="green")
        plt.plot(D, [If(x) for x in D], color="navy")

        plt.xlabel("X")
        plt.ylabel("Y")

        err_galerkin = sum([(If(x) - Ih(x)) ** 2 for x in D]) / len(D)
        err_ls = sum([(If(x) - Ik(x)) ** 2 for x in D]) / len(D)'''

        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита равна {err_galerkin}", end='\n\n')
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Колмогорова-Габора равна {err_ls}", end='\n\n')

        plt.show()

    def draw(self):
        plt.grid(True)
        D = np.linspace(self.m - 3 * self.s, self.m + 3 * self.s, len(self.X))
        g = self.galerkin()
        plt.plot(D, [g(x) for x in D], color="red")
        plt.plot(D, [self.f(x) for x in D], color="navy")
        plt.xlabel("X")
        plt.ylabel("Y")

        err_galerkin = sum([(self.f(x) - g(x)) ** 2 for x in D]) / len(D)
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита равна {err_galerkin}", end='\n\n')

        plt.show()        

class Solver_2D(Solver):
    def __init__(self, f, X, Y, *, acc=0, p=0, silent=False):
        self.f = f

        self.X = X
        self.X.sort()

        self.Y = Y
        self.Y.sort()

        self.mx = np.mean(self.X)
        self.my = np.mean(self.Y)

        self.sx = np.sqrt(np.var(self.X))
        self.sy = np.sqrt(np.var(self.Y))

        self.nx = len(self.X)
        self.ny = len(self.Y)

        self.acc = acc
        self.p = p

        self.silent = silent

    def galerkin(self):
        def acalc(k):
            def a(i, j):
                def hx(kx):
                    return self.X[kx + 1] - self.X[kx]
                
                def hy(ky):
                    return self.Y[ky + 1] - self.Y[ky]
                
                def I(kx, ky):
                    return self.f(self.X[kx], self.Y[ky]) * Solver.He(i, (self.X[kx] - self.mx) / self.sx) * Solver.He(j, (self.Y[ky] - self.my) / self.sy) * np.exp(-(self.X[kx] - self.mx) ** 2 / (2 * self.sx * self.sx)) * np.exp(-(self.Y[ky] - self.my) ** 2 / (2 * self.sy * self.sy))
                
                S = 0
                for kx in range(self.nx - 1):
                    for ky in range(self.ny - 1):
                        S += hx(kx) * hy(ky) * 0.25 * (I(kx, ky) + I(kx + 1, ky) + I(kx, ky + 1) + I(kx + 1, ky + 1))
                
                return S / (2 * np.pi * self.sx * self.sy)
            
            return [[a(i, j) for j in range(k + 1)] for i in range(k + 1)]

        A = []
        L = 0
        if self.p > 0:
            A = acalc(self.p)
            L = self.p
        elif self.acc != 0:
            k = 0
            while k < 6:
                data = acalc(L)
                if max(map(abs, [elem for elem in sum(data, []) if elem not in sum(A, [])])) < self.acc:
                    k += 1
                else:
                    k = 0
                
                A = data
                L += 1
        else:
            raise ValueError("p или acc должны быть положительным числом")
        
        if not self.silent:
            print(f"Разложение производится до {max([len(M) for M in A]) - 1} порядка", end='\n\n')

        def Fest(x, y):
            S = 0
            for i in range(L):
                for j in range(L):
                    S += A[i][j] * Solver.He(i, (x - self.mx) / self.sx) * Solver.He(j, (y - self.my) / self.sy)
            
            return S

        return Fest
    
    def draw(self):
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')

        Dx = np.arange(self.mx - 3 * self.sx, self.mx + 3 * self.sx, 6 * self.sx / (self.nx - 1))    
        Dy = np.arange(self.my - 3 * self.sy, self.my + 3 * self.sy, 6 * self.sy / (self.ny - 1))
        xgrid, ygrid = np.meshgrid(Dx, Dy)
        
        g = self.galerkin()
        axes.plot_surface(xgrid, ygrid, g(xgrid, ygrid), color="red")
        axes.plot_surface(xgrid, ygrid, self.f(xgrid, ygrid), color="navy")

        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")

        err_galerkin = sum([(self.f(x, y) - g(x, y)) ** 2 for x in Dx for y in Dy]) / (len(Dx) * len(Dy))
        print(f"Среднеквадратичная ошибка аппроксимации при использовании полиномов Эрмита равна {err_galerkin}", end='\n\n')

        plt.show()



if __name__ == "__main__":
    np.random.seed(118848)
    def f(x):
        return 10 + x * x - 10 * np.cos(2 * np.pi * x)
    
    err_galerkin = 0
    err_ls = 0
    N = 20
    for i in range(N):
        X = [np.random.normal(0, 0.5) for _ in range(800)]
        s1 = Solver_1D(f, X, p=11, silent=True)
        pr = s1.compare_errors()
        err_galerkin += pr[0]
        err_ls += pr[1]
    
    err_galerkin /= N
    err_ls /= N

    print(err_galerkin, err_ls)

    