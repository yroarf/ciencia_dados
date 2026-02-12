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

# # =============================================
# # Questão 3 – Classificação das variáveis mais importantes
# # =============================================
# print("\n" + "=" * 70)
# print("Questão 3 – Classificação das variáveis principais")
# print("=" * 70)
#
# print("Qualitativas:")
# print("• Ano                  → Qualitativa Ordinal (temporal)")
# print("• UF                   → Qualitativa Nominal (estado)")
# print("• nome_mun             → Qualitativa Nominal (nome do município)")
# print("• Fibra                → Qualitativa Ordinal / Dicotômica (0 ou 1)")
#
# print("\nQuantitativas:")
# print("• IBC                  → Quantitativa Contínua (índice)")
# print("• Cobertura Pop. 4G5G  → Quantitativa Contínua (proporção)")
# print("• Densidade SMP        → Quantitativa Contínua")
# print("• HHI SMP              → Quantitativa Contínua (índice concentração)")
# print("• Densidade SCM        → Quantitativa Contínua")
# print("• HHI SCM              → Quantitativa Contínua")
# print("• Adensamento Estações → Quantitativa Contínua")
# print("• PIB_per_capita       → Quantitativa Contínua (R$)")
# print("• Populacao_2022       → Quantitativa Discreta")
# print("• Area_km2             → Quantitativa Contínua")
# print("• Densidade_hab_km2    → Quantitativa Contínua")
#
#
# # =============================================
# # Questão 4 – Estatística descritiva
# # =============================================
# print("\n" + "=" * 70)
# print("Questão 4 – Estatística descritiva")
# print("=" * 70)
#
# quant_vars = ['IBC', 'Cobertura Pop. 4G5G', 'Densidade SMP', 'HHI SMP',
#               'Densidade SCM', 'HHI SCM', 'Adensamento Estações',
#               'PIB_per_capita', 'Populacao_2022', 'Area_km2', 'Densidade_hab_km2']
#
# desc = df_arquivo[quant_vars].describe(percentiles=[0.25, 0.5, 0.75]).T
# desc['amplitude'] = desc['max'] - desc['min']
# desc['moda'] = df_arquivo[quant_vars].mode().iloc[0]
# desc = desc[['count', 'min', 'max', 'amplitude', 'mean', '50%', 'std', 'moda']]
# desc.columns = ['N', 'Mínimo', 'Máximo', 'Amplitude', 'Média', 'Mediana', 'Desv.Pad', 'Moda']
#
# print("\nEstatísticas das variáveis quantitativas:")
# print(desc.round(2))
#
# print("\nVariáveis qualitativas:")
# print("\nContagem por UF:")
# print(df_arquivo['UF'].value_counts())
#
# print("\nContagem por Fibra:")
# print(df_arquivo['Fibra'].value_counts(dropna=False))
#
# print("\nModa de UF:", df_arquivo['UF'].mode()[0])
#
# # =============================================
# # Questão 5 – Gráficos (3 quantitativas + 3 qualitativas)
# # =============================================
# print("\n" + "=" * 70)
# print("Questão 5 – Gráficos selecionados")
# print("=" * 70)
#
# quant_graf = ['IBC', 'PIB_per_capita', 'Densidade SMP']
# qual_graf = ['UF', 'Fibra', 'Ano']
#
# fig, axes = plt.subplots(3, 2, figsize=(14, 12))
#
# # Quantitativas - Histogramas
# for i, var in enumerate(quant_graf):
#     sns.histplot(data=df_arquivo, x=var, kde=True, ax=axes[i, 0])
#     axes[i, 0].set_title(f'Distribuição de {var}')
#
# # Qualitativas - Barras
# sns.countplot(data=df_arquivo, x='UF', ax=axes[0, 1])
# axes[0, 1].set_title('Quantidade de municípios por UF')
# axes[0, 1].tick_params(axis='x', rotation=45)
#
# sns.countplot(data=df_arquivo, x='Fibra', ax=axes[1, 1])
# axes[1, 1].set_title('Distribuição da variável Fibra (0 = sem, 1 = com)')
#
# sns.countplot(data=df_arquivo, x='Ano', ax=axes[2, 1])
# axes[2, 1].set_title('Distribuição por Ano')
#
# plt.tight_layout()
# plt.show()
#
# print("Interpretações sugeridas:")
# print("- IBC: maioria entre 40 e 70 → conectividade mediana na amostra")
# print("- PIB per capita: distribuição assimétrica positiva → poucos municípios muito ricos")
# print("- Densidade SMP: valores altos concentrados → boa penetração de linhas móveis")
# print("- UF: predomínio de RO e GO (amostra parcial)")
# print("- Fibra: maioria com valor 1 (backhaul presente)")
# print("- Ano: todos 2022 (filtro aplicado)")
#
# # =============================================
# # Questão 6 – Análise de outliers (3 variáveis quantitativas)
# # =============================================
# print("\n" + "=" * 70)
# print("Questão 6 – Análise de outliers")
# print("=" * 70)
#
# out_vars = ['PIB_per_capita', 'IBC', 'Densidade SMP']
#
# for var in out_vars:
#     print(f"\n→ {var}")
#     Q1 = df_arquivo[var].quantile(0.25)
#     Q3 = df_arquivo[var].quantile(0.75)
#     IQR = Q3 - Q1
#     lim_inf = Q1 - 1.5 * IQR
#     lim_sup = Q3 + 1.5 * IQR
#
#     outliers = df_arquivo[(df_arquivo[var] < lim_inf) | (df_arquivo[var] > lim_sup)]
#     n_out = len(outliers)
#
#     print(f"  Limites IQR: [{lim_inf:.2f}, {lim_sup:.2f}]")
#     print(f"  Nº de outliers: {n_out} ({n_out / len(df_arquivo):.2%})")
#
#     if n_out > 0:
#         media_com = df_arquivo[var].mean()
#         mediana_com = df_arquivo[var].median()
#         media_sem = df_arquivo[var][~df_arquivo.index.isin(outliers.index)].mean()
#         mediana_sem = df_arquivo[var][~df_arquivo.index.isin(outliers.index)].median()
#
#         print(f"  Média com outliers:   {media_com:,.2f}")
#         print(f"  Mediana com outliers: {mediana_com:,.2f}")
#         print(f"  Média sem outliers:   {media_sem:,.2f}")
#         print(f"  Mediana sem outliers: {mediana_sem:,.2f}")
#     else:
#         print("  Nenhum outlier detectado pelo método IQR.")
#
# # =============================================
# # Questão 7 – Cruzamento qualitativa × quantitativa (3 combinações)
# # =============================================
# print("\n" + "=" * 70)
# print("Questão 7 – Cruzamento de variáveis")
# print("=" * 70)
#
# combinacoes = [
#     ('UF', 'PIB_per_capita'),
#     ('UF', 'IBC'),
#     ('Fibra', 'Cobertura Pop. 4G5G')
# ]
#
# for qual, quant in combinacoes:
#     print(f"\n→ {qual} × {quant}")
#     tab = df_arquivo.groupby(qual)[quant].agg(['count', 'mean', 'std', 'median']).round(2)
#     print(tab)
#
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(data=df_arquivo, x=qual, y=quant)
#     plt.title(f'{quant} por {qual}')
#     if qual == 'UF':
#         plt.xticks(rotation=45)
#     plt.show()
#
#     print("Interpretação sugerida:")
#     print(f"  {qual} influencia {quant} → verifique diferenças entre estados ou presença de fibra.")
#     print(f"  Exemplo: DF tende a ter PIB per capita e IBC mais altos.\n")