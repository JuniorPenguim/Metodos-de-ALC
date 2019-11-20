from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from random import *
import math
import numpy as np
from numpy import linalg as la
from scipy.stats import norm
from xlrd import *


def carregar_matriz():
    # arquivo = input("Digite o nome do arquivo com extensão:")
    arquivo = 'planilha.xlsx'
    planilha = open_workbook(arquivo).sheet_by_index(0)
    matriz = []

    for linha in range(planilha.nrows):
        matriz.append([])
        for col in range(0, planilha.ncols - 1):
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


def print_matriz(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            print("{0:.4f}  ".format(m[i][j]), end="")

        print()


def print_vetorb(v):
    for i in range(len(v)):
        print("{0:.4f}  ".format(v[i]))


class Questao1:
    def __init__(self):
        self.matriz = carregar_matriz()
        self.vetor_b = carregar_vetorb()
        self.fator_cholesky = []
        self.matriz_l = []
        self.matriz_u = []

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
            pass
        elif letra == 'h':
            self.elimn_gauss()
        elif letra == 'i':
            pass
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

            self.fator_cholesky[0][0] = self.matriz[0][0] ** (1 / 2)  # R 1x1

            self.fator_cholesky[0][1:] = \
                [self.matriz[0][j] / self.fator_cholesky[0][0] for j in range(1, len(self.matriz))]  # R 1xj j = 1...n

            self.fator_cholesky[1][1] = (self.matriz[1][1] - (self.fator_cholesky[0][1] ** 2)) ** (1 / 2)  # R 2x2

            self.fator_cholesky[1][2:] = \
                [(self.matriz[1][j] - (self.fator_cholesky[0][1] * self.fator_cholesky[0][j])) / self.fator_cholesky[1][
                    1]
                 for j in range(2, len(self.matriz))]  # R 2xj j = 3...n

            for i in range(2, len(self.matriz)):
                soma = 0
                for k in range(i):
                    soma += self.fator_cholesky[k][i] ** 2
                self.fator_cholesky[i][i] = (self.matriz[i][i] - soma) ** (1 / 2)

            for i in range(len(self.matriz)):
                for j in range((i + 1), len(self.matriz[i])):
                    soma = 0
                    for k in range(i):
                        soma += self.fator_cholesky[k][i] * self.fator_cholesky[k][j]
                    self.fator_cholesky[i][j] = (self.matriz[i][j] - soma) / self.fator_cholesky[i][i]
            print_matriz(self.fator_cholesky)
        else:
            print("Não é possível calcular o Fator de Cholesky")

    def lu(self):
        if self.verificar_det_submatrizes():

            matriz = np.array(self.matriz)
            matriz_l = self.matriz_l
            matriz_u = self.matriz_u

            for i in range(len(matriz)):
                matriz_l.append([])
                matriz_u.append([])
                for j in range(len(matriz[i])):
                    if i == j:
                        matriz_l[i].append(1)
                    else:
                        matriz_l[i].append(0)
                    matriz_u[i].append(0)

            for j in range(len(matriz)):  # u0j = a0j
                matriz_u[0][j] = matriz[0][j]

            for i in range(1, len(matriz)):  # li0 = ai0/u10
                matriz_l[i][0] = matriz[i][0] / matriz_u[0][0]

            for k in range(1, len(matriz)):  # ukj = akj - soamatorio[0-(k-1), m](lkm * uml) k = 1..n
                soma = 0
                for j in range(k, len(matriz)):
                    for m in range(k - 1):
                        soma += matriz_l[k][m] * matriz_u[m][j]
                    matriz_u[k][j] = matriz[k][j] - soma

            for k in range(2, len(matriz)):  # lik = aik - soamatorio[0-(k-1), m](lim * umk) k = 2..n
                soma = 0
                for i in range(k + 1, len(matriz)):
                    for m in range(k - 1):
                        soma += matriz_l[i][m] * matriz_u[m][k]
                    matriz_l[i][k] = matriz_u[k][k] ** (-1) * (matriz[i][k] - soma)

            # print(np.array(matriz_l))
            # print(np.array(matriz_u))
            self.matriz_l = matriz_l
            self.matriz_u = matriz_u

        else:
            print("Um dos determinantes das submatrizes é igual a zero, "
                  "portanto não é possível prosseguir e encontrar as matrizes L e U")

    def avaliar_criterios(self):
        linhas = self.criterio_linhas()
        colunas = self.criterio_colunas()
        sassenfeld = self.criterio_sassenfeld()
        normas = self.criterio_normas()

        if linhas:
            linhas = "satisfaz o critério das linhas"
        else:
            linhas = "não satisfaz o critério das linhas"

        if colunas:
            colunas = "satisfaz o critério das colunas"
        else:
            colunas = "não satisfaz o critério das colunas"

        if sassenfeld:
            sassenfeld = "satisfaz o critério de Sassenfeld, ou seja," \
                         " a convergência do método de Gauss-Seidel é garantida"
        else:
            sassenfeld = "não satisfaz o critério de Sassenfeld"

        if normas:
            normas = "satisfaz o critério das normas"
        else:
            normas = "não satisfaz o critério das normas"

        print("A matriz A {}, {}, {}, {}.".format(linhas, colunas, sassenfeld, normas))

    def jacobi(self):

        matriz = np.array(self.matriz)
        vetor_b = np.array(self.vetor_b)
        vetor_x = []  # vetor contendo os valores inciais de x
        diagonal = []  # diagonal é a matriz que contém os elementos da diagonal pincipal da matriz A
        m = []  # M contendo todos os elementos da matriz A, menos a diagonal principal
        erro = float(input("Digite o valor da tolerancia:"))

        for i in range(len(matriz)):
            vetor_x.append(float(input("Digite o valor inicial para x{}:".format(i))))

        for i in range(len(matriz)):
            diagonal.append([])
            m.append([])
            for j in range(len(matriz)):
                diagonal[i].append(0)
                m[i].append(0)

        for i in range(len(matriz)):
            for j in range(len(matriz)):
                if i == j:
                    diagonal[i][j] = matriz[i][j]
                else:
                    m[i][j] = matriz[i][j]

        d_inversa = la.inv(np.array(diagonal))  # Retorna a inversa da matriz diagonal
        vetor_x = np.array(vetor_x)

        # Começo das iterações
        itera = 0
        while True:

            vetor_x0 = vetor_x

            vetor_x = d_inversa * vetor_b - d_inversa * m * vetor_x
            itera += 1

            resultado = vetor_x - vetor_x0

            maior_x = 0

            for i in range(len(resultado)):
                for j in range(len(resultado[i])):
                    if resultado[i][j] < 0:
                        if maior_x < -resultado[i][j]:
                            maior_x = float(-resultado[i][j])
                    else:
                        if maior_x < resultado[i][j]:
                            maior_x = float(resultado[i][j])

            if maior_x < erro:
                print('Método parou na {} iteração e obteve como resultado \nX = {}\ne erro = {}'.format
                      (itera, vetor_x.tolist(), maior_x))
                break

    def gauss_seidel(self):

        matrix = self.matriz
        vetor_b = self.vetor_b

        print('Método de Gauss-Seidel')

        tolerancia = float(input("Qual eh a tolerancia? "))

        x = np.zeros(len(matrix))
        k = 0

        solucao_ant = np.zeros(len(matrix))

        for i in solucao_ant:
            i = math.inf

        flag = 0

        while True:

            suma = 0
            k = k + 1
            for r in range(0, len(matrix)):
                suma = 0
                for c in range(0, len(matrix[r]) - 1):
                    if c != r:
                        suma = suma + matrix[r][c] * x[c]
                x[r] = (vetor_b[r] - suma) / matrix[r][r]
                print("x[" + str(r) + "]: " + str(x[r]))

            for j in range(0, len(x)):

                if abs(x[j] - solucao_ant[j]) < tolerancia:
                    flag = 1

                else:
                    flag = 0
            if flag == 1:
                break

            solucao_ant = x.copy()

    def sor(self):
        pass

    def elimn_gauss(self):

        matrix = self.matriz
        vetor_b = self.vetor_b

        m = len(matrix)
        n = len(matrix[0])

        x = np.zeros(m)

        for k in range(0, m):
            for r in range(k + 1, m):
                factor = (matrix[r][k] / matrix[k][k])
                vetor_b[r] = vetor_b[r] - (factor * vetor_b[k])
                for c in range(0, n):
                    matrix[r][c] = matrix[r][c] - (factor * matrix[k][c])

        # substituição pra trás

        x[m - 1] = vetor_b[m - 1] / matrix[m - 1][m - 1]

        print(x[m - 1])

        for r in range(m - 2, -1, -1):
            soma = 0
            for c in range(0, n):
                soma = soma + matrix[r][c] * x[c]
            x[r] = (vetor_b[r] - soma) / matrix[r][r]

        print("Resultado da matriz: ")

        for linha in matrix:
            for elemento in linha:
                print(elemento, end=" ")
            print()

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
            for j in range(0, n - 1):
                sys.stdout.write('%.2f\t' % A_inversa[i][j])
            print()
        print()

        # Número condição de A: fazer norma infinita de linha
        # Norma infinita de linha da matriz A
        maxA = 0
        for i in range(0, m, 1):
            total = 0
            for j in range(0, n - 1, 1):
                total += math.fabs(matrix[i][j])
            if maxA < total:
                maxA = total

        # Norma infinita de linha da matriz A inversa
        maxAI = 0
        for i in range(0, m, 1):
            total = 0
            for j in range(0, n - 1, 1):
                total += math.fabs(A_inversa[i][j])
            if maxAI < total:
                maxAI = total

        # Cálculo do número condição
        num_cond = maxA * maxAI
        print('Número condição:', num_cond)
        print()

    def gradiente_conjugado(self):
        pass

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

    def criterio_linhas(self):
        matriz = self.matriz

        for i in range(len(matriz)):
            soma = 0
            for j in range(len(matriz[i])):
                if i != j:
                    soma += matriz[i][j]
            if abs(soma) >= abs(matriz[i][i]):
                return False

        return True

    def criterio_colunas(self):
        matriz = self.matriz

        for j in range(len(matriz[0])):
            soma = 0
            for i in range(len(matriz)):
                if i != j:
                    soma += matriz[i][j]
            if abs(soma) >= abs(matriz[j][j]):
                return False

        return True

    def criterio_sassenfeld(self):
        matriz = self.matriz
        b = []

        for j in range(len(matriz[0])):
            soma = 0
            for i in range(len(matriz)):
                if i != j:
                    soma += matriz[i][j]  # Ainda necessario ver como fica o somatório aqui
            b.append(soma / abs(matriz[j][j]))

        if max(b) < 1:
            return True
        else:
            return False

    def criterio_normas(self):
        return False

    def verificar_det_submatrizes(self):
        matriz = self.matriz
        d = 0

        for u in range(len(matriz)):
            sub_matriz = []

            for i in range(len(matriz) - d):
                if d > 0:
                    sub_matriz.append(matriz[i][:-d])
                else:
                    sub_matriz.append(matriz[i][:])

            if la.det(sub_matriz) == 0:
                return False
            d += 1

        return True


class Questao2:
    def __init__(self):
        self.vetores = carregar_vetores()

        print(self.vetores)
        print()

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

    def num_vetores_media(self):
        vetores = self.vetores
        cont = 0
        medias = []

        for vetor in vetores:
            cont += 1
            soma = 0
            for item in vetor:
                soma += item
            medias.append(soma / len(vetor))

        print("Recebidos {} vetores.".format(cont))
        for m in range(len(medias)):
            print("media do {}º vetor = {};".format(m + 1, medias[m]))

    def base_ortonormal(self):  # Gram-Schmidt
        vetores = np.array(self.vetores)
        u_vet = [vetores[0] / la.norm(vetores[0])]

        for k in range(1, len(vetores)):
            proj = 0
            for n in range(k):
                proj += (np.dot(vetores[k], u_vet[n])) * u_vet[n]
            y = vetores[k] - proj
            u_vet.append(y / la.norm(y))

        print("Encontrada a base ortonormal:")
        for k in range(len(vetores)):
            print(u_vet[k].tolist())
        print()

    def calcular_angulo_vetores(self):
        i1 = int(input("Escolha um vetor entre os 0-{} possíveis por número:".format(len(self.vetores) - 1)))
        i2 = int(input("Escolha um segundo vetor:"))
        vet1 = np.array(self.vetores[i1])
        vet2 = np.array(self.vetores[i2])
        # print("x =", vet1)
        # print("y =", vet2)
        # print("<x ,y> =", np.dot(vet1, vet2))
        # print("||x|| =", la.norm(vet1))
        # print("||y|| =", la.norm(vet2))

        angulo = np.arccos(np.dot(vet1, vet2) / (la.norm(vet1) * la.norm(vet2)))

        print("O ângulo entre os vetores {} e {} é de {} radianos".format(i1, i2, angulo))

    def calcular_normas_vetores(self):

        i1 = int(input("Escolha um vetor entre os 0-{} possíveis por número:".format(len(self.vetores) - 1)))
        vet1 = np.array(self.vetores[i1])

        def norma1_b(v):
            n1b = 0
            for i in range(len(v)):
                n1b += math.fabs(vet1[i])
            return n1b

        norma_um = norma1_b(vet1)

        print("Norma 1 do vetor é:", norma_um)

        def norma2_b(v):
            n = 0
            for i in range(len(v)):
                n += (math.fabs(vet1[i])) * (math.fabs(vet1[i]))
            return math.sqrt(n)

        norma_dois = norma2_b(vet1)
        print("Norma 2 do vetor é:", round(norma_dois, 2))

        def norma_inf_b(v):
            ninfb = math.fabs(v[0])
            for i in range(len(v)):
                if math.fabs(v[i]) > ninfb:
                    ninfb = (math.fabs(v[i]))
            return ninfb

        norma_inf_i = norma_inf_b(vet1)
        print("Norma Infinita é:", norma_inf_i)

    def produto_interno_vetores(self):
        i1 = int(input("Escolha um vetor entre os 0-{} possíveis por número:".format(len(self.vetores) - 1)))
        i2 = int(input("Escolha um segundo vetor:"))
        vet1 = np.array(self.vetores[i1])
        vet2 = np.array(self.vetores[i2])
        print("x =", vet1)
        print("y =", vet2)

        soma = 0
        for i in range(len(vet1)):
            soma += vet1[i] * vet2[i]

        print("O produto interno entre os vetores {} e {} é de {}".format(i1, i2, soma))

    def calcular_normas_matriz(self):

        def norma_frob_a(a):
            n_fa = 0
            for i in range(len(a)):
                for j in range(len(a[0])):
                    n_fa += (math.fabs(a[i][j])) ** 2
            n_fa = math.sqrt(n_fa)
            return n_fa

            # normas matriciais (cont)

        norma_m = norma_frob_a(self.vetores)
        print(round(norma_m, 2))

        def norma_l(m):
            aux = []

            for i in range(len(m)):
                somatorio = 0
                for j in range(len(m[i])):
                    somatorio += math.fabs(self.vetores[i][j])
                aux.append(somatorio)

            return max(aux)

        norma_l = norma_l(self.vetores)
        print(norma_l)

        def norma_c(m):
            aux = []

            for j in range(len(m[0])):
                somatorio = 0
                for i in range(len(m)):
                    somatorio += math.fabs(self.vetores[i][j])
                aux.append(somatorio)

            return max(aux)

        norma_c = norma_c(self.vetores)
        print(norma_c)


class Questao3:
    def __init__(self):
        self.matriz = carregar_vetores()
        print_matriz(self.matriz)

        letra = input("a) Calcular autovalores.\n"
                      "b) Calcular determinante.\n"
                      )
        if letra == 'a':
            self.autovalores()
        elif letra == 'b':
            self.calcular_det()

    def autovalores(self):

        matrix = self.matriz
        autov = la.eigvals(matrix)

        print(autov)

    def calcular_det(self):

        matriz = self.matriz
        print_matriz(matriz)

        def determinant_recursivo(matriz, total=0):
            # Guarda índices numa lista para referencia recursiva
            indices = list(range(len(matriz)))

            # Quando a matriz for uma 2x2, acaba a recursão e faz o calculo simples da det
            if len(matriz) == 2 and len(matriz[0]) == 2:
                val = matriz[0][0] * matriz[1][1] - matriz[1][0] * matriz[0][1]
                return val

            # define uma sub matriz
            # elimina uma linha e os elementos de cada coluna desta linha
            for fc in indices:

                As = matriz[1:]
                height = len(As)

                for i in range(height):
                    As[i] = As[i][0:fc] + As[i][fc + 1:]

                sign = (-1) ** (fc % 2)  # F)

                # faz o calculo da submatriz de forma recursiva

                sub_det = determinant_recursivo(As)

                # total armezanado da determinante sendo retornado

                total += sign * matriz[0][fc] * sub_det

            return total

        determinante = determinant_recursivo(matriz, total=0)

        print("A determinante da matriz é:", determinante)


class Questao4:
    def __init__(self):
        letra = input("a) Calcular Newton Rhapson para um polinômio de grau 10.\n"
                      "b) Comparar raizes analíticas com Newton Rhapson.\n"
                      )
        if letra == 'a':
            self.calcular_newton_rapshon()
        elif letra == 'b':
            self.calcular_raizes()

    def calcular_newton_rapshon(self):

        coeficientes = []  # vetor contendo os coeficientes
        for i in range(11)[::-1]:
            coef = float(input("Digite o coeficiente para o x^{}:".format(i)))
            coeficientes.append(coef)
        # print("A função ficou com a seguinte cara:\n"
        #       "{}x^10 + {}x^9 + {}x^8 + {}x^7 + {}x^6 + {}x^5 + {}x^4 + {}x^3 + {}x^2 + {}x + {}".format
        #       (coeficientes[0], coeficientes[1], coeficientes[2], coeficientes[3], coeficientes[4],
        #        coeficientes[5], coeficientes[6], coeficientes[7], coeficientes[8], coeficientes[9], coeficientes[10]))

        x0 = float(input("Digite o valor do x incial:"))  # substituir pelo velor do x inicial
        e = float(input("Digite o valor da tolerancia:"))

        i = 0
        x = [x0]
        er = [e + 1.0, e + 1.0]

        while er[0] > e and er[1] > e:
            p1 = self.func(x[i], coeficientes) / self.der(x[i], coeficientes)
            p2 = x[i] - p1

            x.append(p2)
            print(x[i])

            er = self.E(x[i], x[i + 1], self.func)

            i += 1

        return x

    def calcular_raizes(self):

        coeficientes = []  # vetor contendo os coeficientes
        for i in range(4)[::-1]:
            coef = float(input("Digite o coeficiente para o x^{}:".format(i)))
            coeficientes.append(coef)

        possiveis_raizes = []
        raizes_fracionarias = []
        term_indep = int(coeficientes[-1])
        for raiz in range(1, abs(term_indep) + 1):
            if (abs(term_indep) % raiz) == 0:
                possiveis_raizes.append(-raiz)
                possiveis_raizes.append(raiz)
                if raiz / coeficientes[0] not in possiveis_raizes:
                    raizes_fracionarias.append((-raiz) / coeficientes[0])
                    raizes_fracionarias.append(raiz / coeficientes[0])
        possiveis_raizes.extend(raizes_fracionarias)
        # print(possiveis_raizes)

        raizes = []
        for raiz in possiveis_raizes:
            if self.func(raiz, coeficientes) == 0:
                raizes.append(raiz)
                break
        if raizes:

            # Briot Ruffini
            termo = coeficientes[0] * raizes[0]
            coef_d = [termo]
            # print("{} = {} * {}".format(termo, raizes[0], termo / raizes[0]))
            for i in range(1, len(coeficientes) - 1):
                termo += coeficientes[i]
                # print("{} = {} + {}".format(termo, termo - coeficientes[i], coeficientes[i]))
                termo *= raizes[0]
                # print("{} = {} * {}".format(termo, raizes[0], termo / raizes[0]))
                coef_d.append(termo)

            termo += coeficientes[-1]
            # print("{} = {} + {}".format(termo, termo - coeficientes[-1], coeficientes[-1]))

            # Baskara
            raizes.append((-coef_d[1] - ((coef_d[1] ** 2) - (4 * coef_d[0] * coef_d[2])) ** (1 / 2)) / (2 * coef_d[0]))
            raizes.append((-coef_d[1] + ((coef_d[1] ** 2) - (4 * coef_d[0] * coef_d[2])) ** (1 / 2)) / (2 * coef_d[0]))
            print(raizes)
        else:
            print("Não encontrada solução analítica.")

        x0 = float(input("Digite o valor do x incial:"))  # substituir pelo velor do x inicial
        e = float(input("Digite o valor da tolerancia:"))

        i = 0
        x = [x0]
        er = [e + 1.0, e + 1.0]

        while er[0] > e and er[1] > e:
            p1 = self.func(x[i], coeficientes) / self.der(x[i], coeficientes)
            p2 = x[i] - p1

            x.append(p2)
            print(x[i])

            er = self.E(x[i], x[i + 1], self.func)

            i += 1

        print("Raiz encontrada por Newton-Rhapson:", x[-1])

    def func(self, x, coeficientes):
        y = 0
        ex = len(coeficientes) - 1
        for c in coeficientes:
            y += c * (x ** ex)
            ex -= 1

        return y

    def der(self, x, coeficientes):
        y = 0
        ex = len(coeficientes) - 1
        for c in coeficientes:
            if (ex - 1) >= 0:
                y += (ex * c) * (x ** (ex - 1))
            ex -= 1

        return y

    def E(self, x0, x1, func):
        return [abs(func(x1)), abs(x1 - x0)]


class Questao5:
    def __init__(self):

        self.vector = []
        self.vetor_porcentagem = []
        self.matriz = carregar_matriz()
        self.vetor_b = carregar_vetorb()

        print_matriz(self.matriz)
        print_vetorb(self.vetor_b)

        letra = input("a) Construir uma função que substitua cada elemento da matriz pelo seu primeiro dígito.\n"
                      "b) Construir uma função que substitua cada elemento da matriz pelo seu segundo dígito\n"
                      "c) Construir uma função que calcule a estatística de Benford para cada dígito \n"
                      "d) Plotar os resultados obtidos em (c) colocando no mesmo gráfico os valores esperados \n "
                      )
        if letra == 'a':
            self.sub_prim_elemento()
        elif letra == 'b':
            self.sub_seg_elemento()
        elif letra == 'c':
            self.calcular_estat()
        elif letra == 'd':
            self.plotar_result()

    def sub_prim_elemento(self):
        matrix = self.matriz

        matrix_fim = []

        for linha in range(len(matrix)):
            matrix_fim.append([])
            for coluna in range(len(matrix[linha])):
                aux = str(matrix[linha][coluna])
                matrix_fim[linha].append(int(aux[0]))

        print_matriz(matrix_fim)

    def sub_seg_elemento(self):
        matrix = self.matriz
        matrix_fim = []

        for linha in range(len(matrix)):
            matrix_fim.append([])
            for coluna in range(len(matrix[linha])):
                aux = str(matrix[linha][coluna])
                if aux[1] == '.':
                    matrix_fim[linha].append(int(aux[0]))
                else:
                    matrix_fim[linha].append(int(aux[1]))

        print_matriz(matrix_fim)

    def calcular_estat(self):

        matrix = self.matriz
        vector = self.vector
        vetor_porcentagem = self.vetor_porcentagem
        total = 0

        for elemento in range(9):
            vector.append(0)
            vetor_porcentagem.append(0)

        for linha in range(len(self.matriz)):
            for coluna in range(len(self.matriz)):
                total = total + 1
                # for i in range(1, len(vector)+1):
                #     if matrix[linha][coluna] == i:
                #         vector[i-1] += 1
                if matrix[linha][coluna] == 1:
                    vector[0] = vector[0] + 1
                if matrix[linha][coluna] == 2:
                    vector[1] = vector[1] + 1
                if matrix[linha][coluna] == 3:
                    vector[2] = vector[2] + 1
                if matrix[linha][coluna] == 4:
                    vector[3] = vector[3] + 1
                if matrix[linha][coluna] == 5:
                    vector[4] = vector[4] + 1
                if matrix[linha][coluna] == 6:
                    vector[5] = vector[5] + 1
                if matrix[linha][coluna] == 7:
                    vector[6] = vector[6] + 1
                if matrix[linha][coluna] == 8:
                    vector[7] = vector[7] + 1
                if matrix[linha][coluna] == 9:
                    vector[8] = vector[8] + 1

        for elemento in range(9):
            porcentagem = (vector[elemento] / total) * 100
            # print('A porcentagem de números ', elemento, ' é de : ', porcentagem,'%')
            vetor_porcentagem[elemento] = porcentagem
            print('A porcentagem de números ', elemento, 'é de porcentagem  {:.2f}''%'.format(porcentagem))

        self.vector = vector
        self.vetor_porcentagem = vetor_porcentagem

    def plotar_result(self):

        self.calcular_estat()
        vetor_porcentagem = self.vetor_porcentagem

        vector_esperado = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]

        x = np.arange(1, 10)

        plt.plot(x, vetor_porcentagem)
        plt.plot(x, vector_esperado)
        plt.legend(['Frequência Encontrada', 'Frequencia Benford'])
        plt.grid(True)
        plt.xlabel("Frequência para o 1º dígito")
        plt.show()


class Questao6:
    def __init__(self):
        letra = input("a) Calcular Naive Bayes.\n"
                      "b) Calcular Dendrogramação.\n"
                      "c) Calcular método de Érito Marques.\n"
                      )
        if letra == 'a':
            self.naive_bayes()
        elif letra == 'b':
            print("Não ficou pronto")
        elif letra == 'c':
            print("Não ficou pronto")

    def naive_bayes(self):
        # generate 2d classification dataset
        X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
        # summarize
        print(X.shape, y.shape)
        print(X[:5])
        print(y[:5])

        # sorteando os dados em classes
        X_y0 = X[y == 0]
        X_y1 = X[y == 1]
        print(X_y0.shape, X_y1.shape)

        # plt.scatter(X_y0[:, 0], X_y0[:, 1])
        # plt.scatter(X_y1[:, 0], X_y1[:, 1])
        # plt.show()

        # calculate priors
        priory0 = len(X_y0) / len(X)
        priory1 = len(X_y1) / len(X)
        print(priory0, priory1)

        # cria PDFs para y==0
        dist_X1y0 = self.fit_distribution(X_y0[:, 0])
        dist_X2y0 = self.fit_distribution(X_y0[:, 1])
        # cria PDFs para y==1
        distX1y1 = self.fit_distribution(X_y1[:, 0])
        distX2y1 = self.fit_distribution(X_y1[:, 1])

        # classificar uma amostra
        amostra_X, amostra_y = X[0], y[0]

        # calcula a probabilidade de y == 0 na amostra amostra_X
        py0 = self.probabilidade(amostra_X, priory0, dist_X1y0, dist_X2y0)
        # calcula a probabilidade de y == 1 na amostra amostra_X
        py1 = self.probabilidade(amostra_X, priory1, distX1y1, distX2y1)

        print('P(y=0 | {}) = {}'.format(amostra_X, py0 * 100))
        print('P(y=1 | {}) = {}'.format(amostra_X, py1 * 100))
        print('Truth: y = {}'.format(amostra_y))

    def fit_distribution(self, data):
        # estima parâmetros
        media = np.mean(data)
        desvio = np.std(data)
        print(media, desvio)

        # fit distribution
        dist = norm(media, desvio)
        return dist

    def probabilidade(self, x, prior, dist1, dist2):
        return prior * dist1.pdf(x[0]) * dist2.pdf(x[1])


questao = input("Questão 1).\n"
                "Questão 2).\n"
                "Questão 3).\n"
                "Questão 4).\n"
                "Questão 5).\n"
                "Questão 6).\n"
                )
if questao == '1':
    Questao1()
elif questao == '2':
    Questao2()
elif questao == '3':
    Questao3()
elif questao == '4':
    Questao4()
elif questao == '5':
    Questao5()
elif questao == '6':
    Questao6()


