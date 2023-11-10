import pandas as pd
arquivo = pd.read_csv('C:/Users/marco\OneDrive/Área de Trabalho/wine_dataset.csv')
arquivo.head()

#tornando os adjetivos da coluna style em dados numericos
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

#armazenando na variavel "y" os dois tipos de vinho, que serao a caracteristica alvoque ira ser prevista
y = arquivo['style']

#armazenando na variacel "x" as demais variaveis preditoras
x = arquivo.drop('style', axis = 1)

#agora separaremos os dados de teste e de treino, atraves da funcao train_test_split() do pacote sklearn
#destinaremos 30 por cento dos dados para o teste e 70 por cento dos dados para o treino
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)
#print(arquivo.shape, x_treino.shape, x_teste.shape, y_treino.shape, y_teste.shape)

#agora entraremos na parte do algoritmo de classificacao de Machine Learning usando arvores de decisao
#para isso, utilizaremos o algoritmo ExtraTrees, que sera responsavel por criar diversas arvores de decisao
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
#atraves do metodo fit, passaremos ao algoritmo os dados de treino, com a variavel alvo (y) e as variaveis  preditoras(x)
modelo.fit(x_treino, y_treino)
#com o metodoo score, passaremos os dados de teset 
resultado = modelo.score(x_teste, y_teste)
#verificando a acuracia do nosso algoritmo
print("Acurácia:", resultado)

#buscamos uma acuracia superior a 99%. Logo, podemos comecar a testar se nossas predicoes estao condizentes
#com as nossas amostras de dados
#selecionando aleatoriamente 5 amostras:
print(y_teste[400:405])

#realizaremos as previsoes das mesmas amostras com o nosso modelo:
previsoes = modelo.predict(x_teste[400:405])
print(previsoes)