# Passo a Passo de um Projeto de Ciência de Dados
'''Passo 1: Entendimento do Desafio
Passo 2: Entendimento da Área/Empresa
Passo 3: Extração/Obtenção de Dados
Passo 4: Ajuste de Dados (Tratamento/Limpeza)
Passo 5: Análise Exploratória
Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
Passo 7: Interpretação de Resultados'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#Importar a base de dados para o python
tabela = pd.read_csv('advertising.csv')
#analise exploratória -> Entender  como  a sua base de dados está se comportando
#cria o grafico
sns.heatmap(tabela.corr(), annot=True, cmap='Wistia')
#exibe o grafico
plt.show()
#Separando em dados de treino e dados de teste
'y -> Quem você quer prever = vendas'
'x -> o resto da base de dados (quem você vair usar pra fazer a previsão'
y = tabela['Vendas']
x = tabela[['TV','Radio','Jornal']]
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3)
#criar a inteligencia artificial e fazer as previsões
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()
#Treinar a inteligencia artificial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

#testar pra ver qual inteligencia artificial é melhor
previsao_RegressaoLinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoreDecisao = modelo_arvoredecisao.predict(x_teste)
print(r2_score(y_teste, previsao_RegressaoLinear))
print(r2_score(y_teste, previsao_arvoreDecisao))
#Visualização Gráfica das Previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsoes Arvore de Decisao'] = previsao_arvoreDecisao
tabela_auxiliar['Previsoes Regressao Linear'] = previsao_RegressaoLinear
plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()
#Nova previsão
novos = pd.read_csv('novos.csv')
print(novos)
#modelo vencedor foi a Árvore de Decisão
previsao = modelo_arvoredecisao.predict(novos)
print(previsao)



