import math
from numpy.core import double
import numpy
from xlrd import *
from random import *
from numpy import linalg as LA
seed(1)


def carregar_matriz():
    # arquivo = input("Digite o nome do arquivo com extensão:")
    arquivo = 'planilha.xlsx'
    planilha = open_workbook(arquivo).sheet_by_index(0)
    matriz = []

    for linha in range(planilha.nrows):
        matriz.append([])
        for col in range(0, planilha.ncols -1):
            matriz[linha].append(float(planilha.cell_value(linha, col)))

    return matriz


def carregar_vetorb():
    arquivo = 'planilha.xlsx'
    planilha = open_workbook(arquivo).sheet_by_index(0)
    vetor_b = []

    for linha in range(0, planilha.nrows):
        vetor_b.append(planilha.cell_value(linha, planilha.nrows))

    return vetor_b


def carregar_vetores():
    arquivo = 'planilha.xlsx'
    planilha = open_workbook(arquivo).sheet_by_index(0)
    vetores = []

    for vetor in range(0, planilha.nrows):
        vetores.append([])
        for indice in range(planilha.ncols):
            vetores[vetor].append(float(planilha.cell_value(vetor, indice)))

    return vetores


class Exercicio1:
    def __init__(self):
        self.matriz = carregar_matriz()
        self.vetor_b = carregar_vetorb()
        self.fator_cholesky = []

        print_matriz(self.matriz)
        print_vetorb(self.vetor_b)
        letra = input("a) Avaliar se é triangular superior, inferior, diagonal ou nenhuma das três.\n"
                      "b) Calcular o fator de Cholesky.\n"
                      "c) Construir matrizes L e U.\n"
                      "d) Avaliar critérios: linhas, colunas, Sassenfeld e norma.\n"
                      "e) Método de Jacobi.\n"
                      "f) Método de Gauss-Seidel.\n"
                      "g) Método SOR.\n"
                      "h) Eliminação de Gauss.\n"
                      "i) Decomposição SDV.\n"
                      "k) Fatoração QR.\n"
                      "l) Calcular número condição.\n"
                      "m) Gradiente conjugado.\n"
                      )
        if letra == 'a':
            self.identificar_matriz()
        elif letra == 'b':
            self.calcular_fator_cholesky()
        elif letra == 'c':
            self.lu()
        elif letra == 'd':
            self.avaliar_criterios()
        elif letra == 'e':
            self.jacobi()
        elif letra == 'f':
            self.gauss_seidel()
        elif letra == 'g':
            self.sub_prim_elemento()  # subs primeiro elemento!!! trocar dps
        elif letra == 'h':
            self.elimn_gauss()
        elif letra == 'i':
            self.autovalores()  # autovalores!!! trocar dps
        elif letra == 'j':
            self.sub_seg_elemento()  # subs segundo elemento!!! trocar dps
        elif letra == 'k':
            pass
        elif letra == 'l':
            self.num_condicao()
        elif letra == 'm':
            pass

    def identificar_matriz(self):
        superior = self.is_triangular_superior()
        inferior = self.is_triangular_inferior()
        diagonal = self.is_diagonal()

        if superior:
            superior = "é triangular superior"
        else:
            superior = "não é triangular superior"

        if inferior:
            inferior = "é triangular inferior"
        else:
            inferior = "não é triangular inferior"

        if diagonal:
            diagonal = "é diagonal"
        else:
            diagonal = "não é diagonal"
        print("A matriz A {}, {}, {}.".format(superior, inferior, diagonal))

    def calcular_fator_cholesky(self):
        positiva_definida = self.is_positiva_definida()
        if positiva_definida:
            for i in range(len(self.matriz)):
                self.fator_cholesky.append([])
                for j in range(len(self.matriz[i])):
                    self.fator_cholesky[i].append(0)

            self.fator_cholesky[0][0] = self.matriz[0][0]**(1/2)  # R 1x1

            self.fator_cholesky[0][1:] = \
                [self.matriz[0][j]/self.fator_cholesky[0][0] for j in range(1, len(self.matriz))]  # R 1xj j = 1...n

            self.fator_cholesky[1][1] = (self.matriz[1][1]-(self.fator_cholesky[0][1]**2))**(1/2)  # R 2x2

            self.fator_cholesky[1][2:] = \
                [(self.matriz[1][j] - (self.fator_cholesky[0][1]*self.fator_cholesky[0][j])) / self.fator_cholesky[1][1]
                 for j in range(2, len(self.matriz))]  # R 2xj j = 3...n

            for i in range(2, len(self.matriz)):
                soma = 0
                for k in range(i):
                    soma += self.fator_cholesky[k][i]**2
                self.fator_cholesky[i][i] = (self.matriz[i][i] - soma)**(1/2)

            for i in range(len(self.matriz)):
                for j in range((i+1), len(self.matriz[i])):
                    soma = 0
                    for k in range(i):
                        soma += self.fator_cholesky[k][i]*self.fator_cholesky[k][j]
                    self.fator_cholesky[i][j] = (self.matriz[i][j] - soma)/self.fator_cholesky[i][i]
            print_matriz(self.fator_cholesky)
        else:
            print("Não é possível calcular o Fator de Cholesky")

    def lu(self):
        pass

    def avaliar_criterios(self):
        pass

    def jacobi(self):
        pass

    def gauss_seidel(self):

        arquivo = 'planilha.xlsx'
        planilha = open_workbook(arquivo).sheet_by_index(0)

        matrix = []
        vetor_b = []

        matrix = self.matriz
        vetor_b = self.vetor_b

        m = planilha.nrows
        n = planilha.ncols


        print('Método de Gauss-Seidel')

        tolerancia = float(input("Qual eh a tolerancia? "))

        x = numpy.zeros(m)
        k = 0

        solucao_ant = numpy.zeros(m)

        for i in solucao_ant:
            i = math.inf

        flag = 0

        while (1):

            suma = 0
            k = k + 1
            for r in range(0, m):
                suma = 0
                for c in range(0, n-1):
                    if (c != r):
                        suma = suma + matrix[r][c] * x[c]
                x[r] = (vetor_b[r] - suma) / matrix[r][r]
                print("x[" + str(r) + "]: " + str(x[r]))

            for j in range(0, len(x)):

                if (abs(x[j] - solucao_ant[j]) < tolerancia):
                    flag = 1

                else:
                    flag = 0
            if (flag == 1):
                break

            solucao_ant = x.copy()

    def sor(self):
        pass

    def elimn_gauss(self):

        arquivo = 'planilha.xlsx'
        planilha = open_workbook(arquivo).sheet_by_index(0)

        matrix = []
        vetor_b = []

        matrix = self.matriz
        vetor_b = self.vetor_b

        m = planilha.nrows
        n = planilha.ncols

        x = numpy.zeros(m)

        for k in range(0, m):
            for r in range(k + 1, m):
                factor = (matrix[r][k] / matrix[k][k])
                vetor_b[r] = vetor_b[r] - (factor * vetor_b[k])
                for c in range(0, n-1):
                    matrix[r][c] = matrix[r][c] - (factor * matrix[k][c])

        # substituição pra trás

        x[m - 1] = vetor_b[m - 1] / matrix[m - 1][m - 1]

        print(x[m - 1])

        for r in range(m - 2, -1, -1):
            soma = 0
            for c in range(0, n-1):
                soma = soma + matrix[r][c] * x[c]
            x[r] = (vetor_b[r] - soma) / matrix[r][r]

        print("Resultado da matriz: ")

        for linha in matrix:
            for elemento in linha:
                print(elemento, end=" ")
            print("")

        print("Resultado do vetor b: ")

        for elemento in vetor_b:
            print(elemento)

        print("Resultados: ")

        print(x)

    def decom_svd(self):
        pass

    def qr(self):
        pass

    def num_condicao(self):

        from scipy import linalg

        arquivo = 'planilha.xlsx'
        planilha = open_workbook(arquivo).sheet_by_index(0)

        m = planilha.nrows
        n = planilha.ncols

        matrix = self.matriz

        A_inversa = linalg.inv(matrix)
        print('Matriz A inversa:')
        for i in range(0, m):
            for j in range(0, n-1):
                sys.stdout.write('%.2f\t' % A_inversa[i][j])
            print()
        print()

        # Número condição de A: fazer norma infinita de linha
        # Norma infinita de linha da matriz A
        maxA = 0
        for i in range(0, m, 1):
            total = 0
            for j in range(0, n-1, 1):
                total += math.fabs(matrix[i][j])
            if maxA < total:
                maxA = total

        # Norma infinita de linha da matriz A inversa
        maxAI = 0
        for i in range(0, m, 1):
            total = 0
            for j in range(0, n-1, 1):
                total += math.fabs(A_inversa[i][j])
            if maxAI < total:
                maxAI = total

        # Cálculo do número condição
        num_cond = maxA * maxAI
        print('Número condição:', num_cond)
        print()


    def gradiente_conjugado(self):
        pass

    def sub_prim_elemento(self):

        arquivo = 'planilha.xlsx'
        planilha = open_workbook(arquivo).sheet_by_index(0)

        matrix = self.matriz

        matrix_fim = []

        for linha in range(planilha.nrows):
            matrix_fim.append([])
            for coluna in range(planilha.ncols-1):
                aux = str(matrix[linha][coluna])
                matrix_fim[linha].append(int(aux[0]))


        print_matriz(matrix_fim)

    def sub_seg_elemento(self):

        arquivo = 'planilha.xlsx'
        planilha = open_workbook(arquivo).sheet_by_index(0)

        matrix = self.matriz

        matrix_fim = []

        for linha in range(planilha.nrows):
            matrix_fim.append([])
            for coluna in range(planilha.ncols - 1):
                aux = str(matrix[linha][coluna])
                if aux[1] == '.':
                    matrix_fim[linha].append(int(aux[0]))
                else:
                    matrix_fim[linha].append(int(aux[1]))


        print_matriz(matrix_fim)

    def autovalores(self):

        matrix = self.matriz
        autov = []

        autov = LA.eigvals(matrix)

        print(autov)


    def is_diagonal(self):
        for i in range(len(self.matriz)):
            for j in range(len(self.matriz[i])):
                if i != j and self.matriz[i][j] != 0:
                    return False
        return True

    def is_triangular_superior(self):
        for i in range(len(self.matriz)):
            for j in range(len(self.matriz[i])):
                if i > j and self.matriz[i][j] != 0:
                    return False
        return True

    def is_triangular_inferior(self):
        for i in range(len(self.matriz)):
            for j in range(len(self.matriz[i])):
                if i < j and self.matriz[i][j] != 0:
                    return False
        return True

    def is_simetrica(self):
        for i in range(len(self.matriz)):
            for j in range(len(self.matriz[i])):
                if i > j and self.matriz[i][j] != self.matriz[j][i]:
                    return False
        return True

    def is_positiva_definida(self):
        if len(self.matriz) == len(self.matriz[0]):
            nn = True
        else:
            nn = False
        simetrica = self.is_simetrica()

        vet_x = []
        for i in range(len(self.matriz)):
            vet_x.append(randint(1, 1000))

        vet_matriz = []
        for i in range(len(self.matriz)):
            valor = 0
            for j in range(len(self.matriz[i])):
                valor += vet_x[j] * self.matriz[j][i]
            vet_matriz.append(valor)

        valor = 0
        for i in range(len(vet_matriz)):
            valor += vet_matriz[i] * vet_x[i]

        if nn and simetrica and valor > 0:
            return True
        else:
            if not nn:
                print("A matriz A não possui número igual de linhas x colunas:\nL: {} X C: {}.\n"
                      .format(len(self.matriz), len(self.matriz[0])))
            if not simetrica:
                print("A matriz não é simétrica.")
            if valor <= 0:
                print("xT*A*x <= 0.")
            return False


class Exercicio2:
    def __init__(self):
        self.matriz = carregar_vetores()

        letra = input("a) Calcular o número de vetores recebidos e a média de cada linha.\n"
                      "b) Econtrar base ortonormal de n vetores de dimensão n.\n"
                      "c) Calcular o ângulo entre dois vetores.\n"
                      "d) Calcular normas (1, 2, infinito e a norma induzida por uma "
                      "matriz positiva definida) de um vetor.\n"
                      "e) Calcular o produto interno de dois vetores.\n"
                      "f) Calcular as normas (linha, coluna e de Frobenius) de uma matriz.\n"
                      )
        if letra == 'a':
            self.num_vetores_media()
        elif letra == 'b':
            self.base_ortonormal()
        elif letra == 'c':
            self.calcular_angulo_vetores()
        elif letra == 'd':
            self.calcular_normas_vetores()
        elif letra == 'e':
            self.produto_interno_vetores()
        elif letra == 'f':
            self.calcular_normas_matriz()


class Exercicio3:
    def __init__(self):
        self.matriz = carregar_vetores()

        letra = input("a) Calcular autovalores.\n"
                      "b) Calcular determinante.\n"
                      )
        if letra == 'a':
            self.calcular_autovalores()
        elif letra == 'b':
            self.calcular_det()


class Exercicio4:
    def __init__(self):
        self.matriz = carregar_vetores()

        letra = input("a) Calcular Newton Rhapson para um polinômio de grau 10.\n"
                      "b) Comparar raizes analíticas com Newton Rhapson.\n"
                      )
        if letra == 'a':
            self.calcular_newton_rapshon()
        elif letra == 'b':
            self.calcular_raizes()


class Exercicio5:
    def __init__(self):
        self.matriz = carregar_vetores()

        letra = input("a) Calcular o número de vetores recebidos e a média de cada linha.\n"
                      "b) Econtrar base ortonormal de n vetores de dimensão n.\n"
                      "c) Calcular o ângulo entre dois vetores.\n"
                      "d) Calcular normas (1, 2, infinito e a norma induzida por uma "
                      "matriz positiva definida) de um vetor.\n"
                      "e) Calcular o produto interno de dois vetores.\n"
                      )
        if letra == 'a':
            self.num_vetores_media()
        elif letra == 'b':
            self.base_ortonormal()
        elif letra == 'c':
            self.calcular_angulo_vetores()
        elif letra == 'd':
            self.calcular_normas_vetores()
        elif letra == 'e':
            self.produto_interno_vetores()


class Exercicio6:
    def __init__(self):
        self.matriz = carregar_vetores()

        letra = input("a) Calcular o número de vetores recebidos e a média de cada linha.\n"
                      "b) Econtrar base ortonormal de n vetores de dimensão n.\n"
                      "c) Calcular o ângulo entre dois vetores.\n"
                      )
        if letra == 'a':
            self.num_vetores_media()
        elif letra == 'b':
            self.base_ortonormal()
        elif letra == 'c':
            self.calcular_angulo_vetores()


# MÉTODOS=>

# zero das funções

def bissecao(func, a, b, e):
    while (abs(a - b) > e) and (abs(func((a + b) / 2)) > e):
        x = (a + b)/2

        print("f(x) =", func(x))
        print("f(a) =", func(a))

        if (func(x) * func(a)) < 0:
            b = x
        else:
            a = x

        print("x =", x)
        print()
        print("[a, b] =", [a, b])
        print("|(", a, ") - (", b, ")| =", abs(a - b), ">", e)
        print()
        print("|f( ((", a, ") + (", b, ")) / 2 )| =", abs(func((a + b) / 2)), ">", e)

    return x, [a, b]


def falsaPosi(func, a, b, e):
    def ER(x0, x1):
        return abs(x0 - x1) / abs(x1)

    i = 1
    x = [0]
    er = 1

    while er > e:
        p1 = (a * func(b))
        p2 = (b * func(a))
        p3 = p1 - p2
        p4 = (func(b) - func(a))
        p5 = p3 / p4

        print("p1 = a * func(b) =", p1)
        print("f(b) = ", func(b))
        print("p2 = b * func(a) =", p2)
        print("f(a) = ", func(a))
        print("p3 = p1 - p2 =", p3)
        print("p4 = func(b) - func(a) =", p4)
        print("p5 = p3 / p4 =", p5)

        x.append(p5)
        print(x)
        er = ER(x[i - 1], x[i])

        if func(x) < 0:
            a = x
        else:
            b = x

        i += 1


    return x


def newton_raphson(func, der, x0, e):
    def E(x0, x1, func):
        return [abs(func(x1)), abs(x1 - x0)]

    i = 0
    x = [x0]
    er = [e+1, e+1]

    while er[0] > e and er[1] > e:
        p1 = func(x[i]) / der(x[i])
        p2 = x[i] - p1

        print("p1 =", p1)
        print("p2 = ", p2)

        x.append(p2)
        print(x)

        er = E(x[i], x[i+1], func)

        i += 1

    return x


# segunda ordem

def tangentes(qtd, a, b, func, y0):
    h = (b - a)/qtd
    x = []
    y = []
    n = qtd+1

    x.append(a)

    for i in range(1, n):
        x.append(x[i-1] + h)

    y.append(y0)

    for i in range(1, n):
        y.append(y[i-1] + (h *func(y[i-1])))

    # print(y)

    return y


def rungeKutta2(qtd, a, b, func, y0):
    h = (b - a)/qtd
    x = []
    y = []
    u = 0
    n = qtd+1

    x.append(a)

    for i in range(1, n):
        x.append(x[i-1] + h)

    y.append(y0)

    for i in range(1, n):
        u = y[i-1] + (h * func(y[i-1]))
        y.append(y[i-1] + ((h/2) * (func(y[i-1]) + func(u))))

    # print(y)

    return y


def rungeKutta4(qtd, a, b, func, y0):
    h = (b - a)/qtd
    x = []
    y = []
    u = 0
    n = qtd+1

    x.append(a)

    for i in range(1, n):
        x.append(x[i-1] + h)

    y.append(y0)

    for i in range(1, n):
        k1 = h * func(y[i-1])
        k2 = h * func(y[i-1] + (k1/2))
        k3 = h * func(y[i-1] + (k2/2))
        k4 = h * func(y[i-1] + k3)
        y.append(y[i-1] + ((1/6) * (k1 + (2*k2) + (2*k3) + k4)))

    # print(y)

    return y


# integrais

def simp38(qtd, a, b, func):
    h = (b - a)/qtd
    x = []
    y = []
    r = []
    rT = 0
    n = qtd+1

    x.append(a)

    for i in range(1, n):
        x.append(x[i-1] + h)

    # print(x)

    for i in range(n):
        y.append(func(x[i]))

    # print(y)

    for i in range(n):
        if i == 0 or i == n-1:
            c = 1
        elif (i%3) == 0:
            c = 2
        else:
            c = 3

        r.append(c * y[i])
        rT += r[i]
        print(c, 'x', y[i], '=', r[i])
        print(rT)


    rT = (3 * h * rT) / 8

    return rT


def simp13(qtd, a, b, func):
    h = (b - a)/qtd
    x = []
    y = []
    r = []
    rT = 0
    n = qtd+1

    x.append(a)

    for i in range(1, n):
        x.append(x[i-1] + h)

    # print(x)

    for i in range(n):
        y.append(func(x[i]))

    # print(y)

    for i in range(n):
        if i == 0 or i == n-1:
            c = 1
        elif (i%2) == 0:
            c = 2
        else:
            c = 4

        r.append(c * y[i])
        rT += r[i]
        print(c, 'x', y[i], '=', r[i])
        print(rT)


    rT = (h * rT) / 3

    return rT


def trapeziosComp(qtd, a, b, func):
    h = (b - a)/qtd
    x = []
    y = []
    r = []
    rT = 0
    n = qtd+1

    x.append(a)

    for i in range(1, n):
        x.append(x[i-1] + h)

    # print(x)

    for i in range(n):
        y.append(func(x[i]))

    # print(y)

    for i in range(n):
        if i == 0 or i == n-1:
            c = 1
        else:
            c = 2

        r.append(c * y[i])
        rT += r[i]
        print(c, 'x', y[i], '=', r[i])
        print(rT)


    rT = (h * rT) / 2

    return rT


# MMQ
def MMQ(x, y):
    pass


def print_matriz(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            print("{0:.4f}  ".format(m[i][j]), end="")

        print()


def print_vetorb(v):
    for i in range(len(v)):
        print("{0:.4f}  ".format(v[i]))


questao = input("Exercício 1).\n"
                "Exercício 2).\n"
                "Exercício 3).\n"
                "Exercício 4)."
                "Exercício 5).\n"
                "Exercício 6).\n"
                )
if questao == '1':
    Exercicio1()
elif questao == '2':
    Exercicio2()
elif questao == '3':
    Exercicio3()
elif questao == '4':
    Exercicio4()
elif questao == '5':
    Exercicio5()
elif questao == '6':
    Exercicio6()


