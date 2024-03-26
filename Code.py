import numpy as np
from numpy.linalg import matrix_power
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
A = [[-1, 1/2, 0],
    [1/2, -1, 1/2],
    [0, 1/3, -1]]

b = [-1/2, 0, -1/3] # resultados

estat1 = np.linalg.solve(A,b) # definicao das probabilidades de absorcao
print(estat1)
ajuste = [1,1,1]
absorcao_final = ajuste - estat1
resp1 = absorcao_final[0]

print(absorcao_final)
print("A probabilidade da mosca escapar pela janela sem ser pega pela aranha é",round(resp1,2))
B = [[-1, 2, -1],
    [2, -1, 0],
    [0, -1, -3]]

r =[2,2,3] # resultados

estat2 = np.linalg.solve(B,r)
print(estat2)
print(estat2[0]) # Tempo medio gasto saindo da maca
print(estat2[1]) # Tempo medio gasto saindo da cadeira
C = [[1, -1, 0],
     [-1/2, 1, -1/2],
     [0, -1/2, 1]]

r2 = [1,1,1] # resultados 

estat3 = np.linalg.solve(C,r2) 

print(estat3)
print(estat3[0])               # Tempo medio para sair pela janela
p = 0.14 # probabilidade do exercicio (a)
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o numero de simulacoes
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0, 0.3])
plt.legend()
plt.show(True)
resultado = ((0.51/0.49)**(6)-(0.51/0.49)**(16))/(1-(0.51/0.49)**16)
Prob = 1 - resultado

print(Prob)
p = 0.3 # probabilidade do exercicio (a)
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0.2,0.4])
plt.legend()
plt.show(True) 
resultado1 = ((0.6/0.4)**(6)-(0.6/0.4)**(16))/(1-(0.6/0.4)**16) # Simulnado probabilidades e fortuna iniciais diferentes
Prob1 = 1 - resultado1

resultado2 = ((0.55/0.45)**(5)-(0.55/0.45)**(15))/(1-(0.55/0.45)**15) # Simulnado probabilidades e fortuna iniciais diferentes
Prob2 = 1 - resultado2

print(Prob1)
print(Prob2)
p = 0.015 # probabilidade do exercicio simualado(resultado1)
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0,0.1])
plt.legend()
plt.show(True) 
p = 0.08 # probabilidade do exercicio simulado(resultado2)
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0,0.2])
plt.legend()
plt.show(True) 
E = [[1, -1],
     [-1, 2]]

r3 = [2,2] # resultados  

estat4 = np.linalg.solve(E,r3)
print(estat4)
print(estat4[0])
p = 0.2 
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0,1])
plt.legend()
plt.show(True)
p = 0.4
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0,1])
plt.legend()
plt.show(True) 
p = 0.6
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0,1])
plt.legend()
plt.show(True)
p = 0.8
vp = [] # lista que armazena a fração de ocorrências em função do número de simulações nsim
vsim = [] # armazena o número de simulações
Nmax = 5000 # numero maximo de simulacoes
for nsim in np.arange(0,Nmax,10):
    num = 0 
    for i in range(1,nsim):
        if(np.random.uniform() < p):
            num = num + 1        
    vp.append(num/nsim)
    vsim.append(nsim)

plt.figure(figsize=(10,8))
plt.plot(vsim, vp, linestyle='-', color="blue", linewidth=2,label = 'Valor simulado')
plt.axhline(y=p, color='r', linestyle='--', label = 'Valor teorico')
plt.ylabel("Probabilidades", fontsize=20)
plt.xlabel("Simulações", fontsize=20)
plt.xlim([0.0, Nmax])
plt.ylim([0,1])
plt.legend()
plt.show(True) 
y = np.array([[0,3/6,0,1/6,2/6],[1,0,0,0,0],[0,0,0,0,1],[1/5,0,0,0,4/5],[2/12,0,6/12,4/12,0]])
eigvals, eigvecs = np.linalg.eig(y.T)
eigvec1 = eigvecs[:,np.isclose(eigvals, 1)]
eigvec1 = eigvec1[:,0]
stationary = eigvec1 / eigvec1.sum()
stationary = stationary.real
for i in range(0,5):
    print(i+1, ':', stationary[i])
    #valor simulado
vaposta=[1,2,3,4,5]
vp = [0.186815, 0.093186, 0.187762, 0.156839, 0.375398]
plt.plot(vaposta,vp, label = 'Simul. 1M passos')
plt.xlabel('Estados', fontsize = 15)
plt.ylabel('Valor de probabilidade', fontsize = 15)

vp1 = [0.223, 0.115, 0.166, 0.147, 0.349]
plt.plot(vaposta,vp1, label = 'Simul. mil passos')

#valor teorico
#prob.estacionaria
vpt = [0.18749999999999983, 0.09374999999999975, 0.18750000000000008,0.15625000000000003, 0.3750000000000003]
vapostat = [1,2,3,4,5] #aposta
plt.plot(vapostat,vpt,label='Valor teorico')
plt.legend(fontsize=15)
plt.show(True)
H = nx.read_gml("lesmis.gml")
X = nx.to_numpy_matrix(H) 
print(X)
n = X.shape[0] # armazena o numero de linhas
Z = np.zeros((n,n)) # matriz de zeros NxN
for i in range(0,n): 
    for j in range(0,n): 
        if(np.sum(X[i,:]) > 0): 
            Z[i,j] = X[i,j]/np.sum(X[i,:]) # se tiver relacionamento atribui probabilidade 
        else:
            Z[i,j] = 1/n # caso contrario atribui a mesma probabilidade para os relacionamentos
print(Z)
G = np.zeros((n,n))
alpha = 0.85 
for i in range(0,n):
    for j in range(0,n):
        G[i,j] = alpha*Z[i,j] + (1-alpha)/n
print(G)
eigvals, eigvecs = np.linalg.eig(G.T)
eigvec1 = eigvecs[:,np.isclose(eigvals, 1)]
eigvec1 = eigvec1[:,0]
stationary = eigvec1 / eigvec1.sum()
stationary = stationary.real

for i in range(0,n):
    print(i+1, ':', stationary[i])
options = {
    'node_color': 'blue',
    'node_size': 1000,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 20,
}

npos=nx.circular_layout(H)
s = stationary*10000
fig = plt.figure(figsize=(12, 12))
nx.draw(H,pos = npos, with_labels=True, node_size = s, arrows=True)
plt.draw()
plt.show()
passos = 10000
v_inicio, ini = 0 , 0
qtd = np.zeros(len(Z)) # qtde de visitas em cada vertice
caminhada , v_vizinho = [] , []
for i in range(2):
    qtd[ini] += 1
    caminhada +=[ini] # armazena os vertices visitados
    for j in range(0,passos):
        v_vizinho = [] 
        for k in range(0,len(Z[0])):
            if Z[ini,k]  > 0: # verifica relacao
                v_vizinho += [k] # armazena vertices vizinhos
        passo = np.random.choice(v_vizinho) # escolha aleatoria do proximo passo
        qtd[passo] += 1 
        caminhada += [passo] 
        ini = caminhada[-1] 
        v_vizinho.clear()
    v_inicio += 1 
    ini = v_inicio 

c = []
for m in range(len(qtd)):
    l = [m/sum(qtd)]
    c.append(l)
a = np.reshape(c, (77, 1)) # caminhada aleatoria
b = np.reshape(stationary, (77, 1)) # pagerank

plt.title('PageRank x Número de Visitas')
plt.plot(a,b)
plt.xlabel("Número de Visitas")
plt.ylabel("Page Rank")
plt.show()
