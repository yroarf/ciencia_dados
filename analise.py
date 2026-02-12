import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurações visuais
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
pd.set_option('display.float_format', '{:.2f}'.format)

# =============================================
#  Leitura da Base
# =============================================
df_arquivo = pd.read_csv("df_ibc_pib_pop.csv", sep=",", encoding="utf-8")



numeric_cols = ['IBC', 'Cobertura Pop. 4G5G', 'Densidade SMP', 'HHI SMP',
                'Densidade SCM', 'HHI SCM', 'Adensamento Estações', 'Fibra',
                'PIB_per_capita', 'Populacao_2022', 'Area_km2', 'Densidade_hab_km2']
lista_descricao = []

for col in numeric_cols:
    if col in df_arquivo.columns:
        df_arquivo[col] = pd.to_numeric(df_arquivo[col], errors='coerce')
        serie_descricao = df_arquivo[col].describe()
        df_desc = pd.DataFrame(serie_descricao).T

        df_desc.index = [col]  # nome da variável na linha
        lista_descricao.append(df_desc)

# Junta todos em um único DataFrame
df_estatisticas = pd.concat(lista_descricao)

# Renomeia o índice para ficar mais bonito
df_estatisticas.index.name = 'Variável'

# Reordena colunas (opcional)
colunas_ordem = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
df_estatisticas = df_estatisticas[colunas_ordem]
df_estatisticas = df_estatisticas.rename(columns={
    'count': 'N',
    'mean': 'Média',
    'std': 'Desvio Padrão',
    'min': 'Mínimo',
    '25%': '1º Quartil (25%)',
    '50%': 'Mediana',
    '75%': '3º Quartil (75%)',
    'max': 'Máximo',
    'amplitude': 'Amplitude',
})

# Cálculos que não estão na função describe()

# Moda (valor mais frequente)
moda = df_arquivo[numeric_cols].mode().iloc[0]  # pega a primeira moda (se houver múltiplas)
df_estatisticas['Moda'] = moda.round(2)
print(df_estatisticas['Moda'])

# Variância
variancia = df_arquivo[numeric_cols].var()
df_estatisticas['Variância'] = variancia.round(2)

# Amplitude (já calculada anteriormente, mas garantindo ordem)
df_estatisticas['Amplitude'] = df_estatisticas['Máximo'] - df_estatisticas['Mínimo']
df_estatisticas['Amplitude'] = df_estatisticas['Amplitude'].round(2)


# Reordenar as colunas finais (ordem desejada)

colunas_finais = [
    'N', 'Média', 'Variância', 'Desvio Padrão', 'Mínimo', '1º Quartil (25%)',
    'Mediana', '3º Quartil (75%)', 'Máximo', 'Amplitude', 'Moda'
]

df_estatisticas = df_estatisticas[colunas_finais].round(2)
df_estatisticas.to_csv("describe_todas_variaveis_completo.csv", sep=';', encoding='utf-8-sig')

