"""
    Atividade de Fixação 02
    Técnicas de Programação - Biblioteca Pandas
    ✦ Thaís de Souza Marins ✦

Nessa atividade foi realizada a análise exploratória dos dados do Spotify.

A base de dados utilizada foi retirada da plataforma Kaggle.
Está disponível em: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?utm_source=ActiveCampaign&utm_medium=email&utm_content=%237DaysOfCode+-+Machine+Learning+1%2F7%3A+Coleta+de+dados+e+An%C3%A1lise+Explorat%C3%B3ria&utm_campaign=%5BAlura+%237Days+Of+Code%5D%28Js+e+DOM+-+3%C2%AA+Ed+%29+1%2F7)
____________________________________________________________________________________________________________________"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importando os ignorando a coluna 'Unnamed: 0'
dados = pd.read_csv('dataset.csv', index_col=0)
print(dados.head())

print(f'\nNúmero de linhas: {dados.shape[0]} \nNúmero de colunas:{dados.shape[1]}')

print(f'\nInfo:\n{dados.info()}')

# Estatísticas variáveis numéricas
print(f'Estatísticas variáveis numéricas:\n {dados.describe()}')

# Estatísticas variáveis categóricas
print(dados.describe(include=['O']))

#Valores Faltantes
print(dados.isnull().sum())

#Como resolver o problema de valores faltantes?
#Como não há muitos valores nulos, estes serão removidos da base de dados
dados = dados.dropna()
print(f'\nValores Faltantes Tratados:\n{dados.isnull().sum()}')

#Há valores duplicados na base de dados?
print(f'\nValores Duplicados:\n{dados.duplicated().sum()}')

#Removendo os dados duplicados
dados = dados.drop_duplicates()
print(f'\nValores Duplicados:\n{dados.duplicated().sum()}')

print(f'Colunas:\n {dados.columns}')

"""_______________________________________________________________________________________________________________________

### Explorando os dados:
"""

# Contagem de artistas únicos
print(dados['artists'].unique().shape)
print(dados['artists'].value_counts())

#Quais os 10 artistas mais ouvidos?
top_artistas = dados.groupby('artists').count().sort_values(by='popularity', ascending=False)['popularity'][:10]
top_artistas = top_artistas[::-1]
ax = top_artistas.plot.barh(color='darkviolet')
plt.xlabel('Popularidade')
plt.ylabel('Artistas')
plt.title('Top 10 artistas mais populares')

for i, v in enumerate(top_artistas):
    ax.text(v+1, i, str(v), va='center', fontsize=8, color='black')

plt.show()

#Quais os 10 artistas mais populares?
top_musicas = dados.groupby('track_name').count().sort_values(by='popularity', ascending=False)['popularity'][:10]
top_musicas = top_musicas[::-1]
ax = top_musicas.plot.barh(color="fuchsia")
plt.xlabel('Popularidade')
plt.ylabel('Músicas')
plt.title('Top 10 músicas mais populares')

plt.show()

#Quais as músicas mais longas?
musicas_longas = dados[['track_name', 'duration_ms']].sort_values(by='duration_ms', ascending=False)[:5]
musicas_longas = musicas_longas[::-1]

ax = musicas_longas.plot.barh(y='duration_ms', x='track_name', color='hotpink', legend=False)
plt.xlabel('Duração (ms)')
plt.ylabel('Música')
plt.title('Músicas Mais Longas')

for i, v in enumerate(musicas_longas['duration_ms']):
    ax.text(v-1000009, i, f"{v/60000:.2f} min", va='center', fontsize=8, color='black')

plt.show()

musica_mais_longa = dados[dados['duration_ms'] == dados['duration_ms'].max()]
musica_mais_curta = dados[dados['duration_ms'] == dados['duration_ms'].min()]

#Conversao
duracao_mais_longa_min = musica_mais_longa['duration_ms'] / 60000
duracao_mais_curta_min = musica_mais_curta['duration_ms'] / 60000

print(f'Música mais longa: {musica_mais_longa["track_name"].values[0]} - {duracao_mais_longa_min.values[0]:.2f} minutos')
print(f'Música mais curta: {musica_mais_curta["track_name"].values[0]} - {duracao_mais_curta_min.values[0]:.2f} minutos')

#Criando uma cópia do dataframe para fazer alterações preservando o original
dados_copia = dados.copy()

#Sabendo que 1 minuto tem 60.000 milissegundos será criada uma nova coluna com essa conversão
dados_copia['duration_min'] = (dados_copia['duration_ms'] / 60000).round(2)
print(dados_copia.shape)

# Qual a música mais popular do Elvis? S2
elvis_musicas = dados[dados['artists'].str.contains('Elvis Presley')]
indice = elvis_musicas['popularity'].idxmax()
musica_mais_popular = elvis_musicas.loc[indice]
nome_musica_mais_popular = musica_mais_popular['track_name']

print(f'A música mais popular de Elvis Presley é: "{nome_musica_mais_popular}"')

#Quais os artistas com músicas mais dançantes?
top_artistas_dancantes = dados[['danceability','track_name', 'artists']].sort_values(by='danceability', ascending=False)[:10]
top_artistas_dancantes

# Criando 5 categorias para determinar a faixas de popularidade
dados_copia['popularity_range'] = pd.qcut(dados_copia['popularity'], q=5, labels=['Muito Baixa', 'Baixa', 'Moderada', 'Alta', 'Muito Alta'])
print(dados_copia['popularity_range'])

contagem_popularidade = dados_copia['popularity_range'].value_counts()

ordem_faixas = ['Muito Baixa', 'Baixa', 'Moderada', 'Alta', 'Muito Alta']
contagem_popularidade = contagem_popularidade.loc[ordem_faixas]

colors = ['#FF00FF', '#FF69B4', '#DDA0DD', '#8A2BE2', '#800080']

plt.figure(figsize=(8, 8))
plt.pie(contagem_popularidade, labels=contagem_popularidade.index, autopct='%1.2f%%', colors=colors)
plt.title('Distribuição de músicas por categorias de popularidade')
plt.show()

# Principais gêneros das músicas mais populares
musicas_mais_populares = dados.query('popularity > 98')
contagem_genero = musicas_mais_populares['track_genre'].value_counts()

plt.figure(figsize=(12, 6))
contagem_genero.plot(kind='bar', color='purple')
plt.xlabel('Gênero Musical')
plt.ylabel('Contagem de Músicas')
plt.title('Distribuição de gêneros musicais por popularidade')
plt.xticks(rotation=45)
plt.show()

#Qual a música com a maior valencia? (Mais feliz)
max_valence = dados[dados['valence'] == dados['valence'].max()]
print(max_valence[['track_name', 'artists', 'valence']])

#Distribuição da valência
plt.figure(figsize=(8, 6))
plt.hist(dados['valence'], bins=20, edgecolor='white', color='darkviolet')
plt.xlabel('Valência')
#plt.ylabel('Frequência')
plt.title('Distribuição da valência')
plt.show()

#Há muitas músicas com letras explícitas?
musicas_explicitas = dados[dados['explicit'] == 1].shape[0]
músicas_implícitas = dados[dados['explicit'] == 0].shape[0]

porcentagem_explicitas = (musicas_explicitas / len(dados)) * 100
porcentagem_implicitas = (músicas_implícitas / len(dados)) * 100

print(f'Porcentagem de músicas com letra explícita: {porcentagem_explicitas:.2f}%')
print(f'Porcentagem de músicas com letra implícita: {porcentagem_implicitas:.2f}%')

"""___________________________________________________________________________________________________

## Correlação de variaveis
"""

#Hipóteses
corr_energy_loudness = dados['energy'].corr(dados['loudness'])
corr_energy_val = dados['energy'].corr(dados['valence'])
corr_val_loudness =  dados['valence'].corr(dados['loudness'])
corr_tempo_pop =  dados['danceability'].corr(dados['popularity'])

print(f'\nCorrelação entre Energia e Volume: {corr_energy_loudness:.2f}')
print(f'\nCorrelação entre Energia e Valência: {corr_energy_val :.2f}')
print(f'\nCorrelação entre Valência e Volume: {corr_val_loudness:.2f}')
print(f'\nCorrelação entre Tempo e Popularidade: {corr_tempo_pop:.2f}')

#Estatísticas:
dados_copia['energy_media'] = dados['energy'].mean()
dados_copia['loudness_mediana'] = dados['loudness'].median()
dados_copia['valence_maximo'] = dados['valence'].max()

print(dados_copia)

#dados_filtrados = dados[(dados['energy'] > 0.5) & (dados['valence'] > 0.5) & (dados['danceability'] > 0.5)]
#dados_filtrados

def coluna_feliz(valencia):
    if valencia > 0.5:
        return 'Feliz'
    else:
        return 'Triste'

dados_copia['coluna_feliz'] = dados['valence'].apply(coluna_feliz)
dados_copia.rename(columns={'coluna_feliz': 'mood_category'}, inplace=True)

cont_feliz = (dados_copia['mood_category'] == 'Feliz').sum()
cont_triste = (dados_copia['mood_category'] == 'Triste').sum()

print(f'Feliz: {cont_feliz}')
print(f'Triste: {cont_triste}')

dados_copia.to_csv('dados_modificados.csv', index=False)