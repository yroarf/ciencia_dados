import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from scipy import stats
import warnings

# Configurações visuais
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
pd.set_option('display.float_format', '{:.2f}'.format)

# ===========================================================
#  >>>>>>>>>>>>>>>>> Bloco: Leitura da Base <<<<<<<<<<<<<<<<<
# ===========================================================

df_arquivo = pd.read_csv("df_ibc_pib_pop.csv", sep=",", encoding="utf-8")

# =================================================================
#  >>>>>>>>>>>>>>>>> Bloco: Cálculos Estatísticos <<<<<<<<<<<<<<<<<
#  >>>>>>>>>>>>>>>>>            Questão 3         <<<<<<<<<<<<<<<<<
# =================================================================

# Garantir que as variáveis típo numéricas não sejam carregadas como texto

numeric_cols = [
    'IBC', 'Cobertura Pop. 4G5G', 'Densidade SMP', 'HHI SMP',
    'Densidade SCM', 'HHI SCM', 'Adensamento Estações', 'Fibra',
    'PIB_per_capita', 'Populacao_2022', 'Area_km2', 'Densidade_hab_km2'
]

for col in numeric_cols:
    if col in df_arquivo.columns:
        df_arquivo[col] = pd.to_numeric(df_arquivo[col], errors='coerce')

# Variáveis qualitativas (categóricas)
qualitative_vars = ['UF', 'Ano', 'Fibra']

# Variáveis quantitativas (contínuas ou discretas)
quantitative_vars = [
    'IBC', 'Cobertura Pop. 4G5G', 'Densidade SMP', 'HHI SMP',
    'Densidade SCM', 'HHI SCM', 'Adensamento Estações',
    'PIB_per_capita', 'Populacao_2022', 'Area_km2', 'Densidade_hab_km2'
]

# Remover colunas que não existem ou estão totalmente vazias
quantitative_vars = [col for col in quantitative_vars if col in df_arquivo.columns and df_arquivo[col].notna().sum() > 0]

print("═" * 70)
print("ANÁLISE ESTATÍSTICA DESCRITIVA")
print("═" * 70)
print(f"Total de observações: {len(df_arquivo):,}")
print(f"Quantidade de municípios únicos: {df_arquivo['nome_mun'].nunique():,}")
print(f"Quantidade de UFs presentes: {df_arquivo['UF'].nunique()}")
print()

# =============================================
# 2. Variáveis Quantitativas
# =============================================
print(" VARIÁVEIS QUANTITATIVAS ".center(70, "═"))
print()

for var in quantitative_vars:
    serie = df_arquivo[var].dropna()
    if len(serie) == 0:
        continue

    n = len(serie)
    minimo = serie.min()
    maximo = serie.max()
    amplitude = maximo - minimo
    media = serie.mean()
    mediana = serie.median()
    variancia = serie.var()
    desvio_padrao = serie.std()

    # Moda (pode haver múltiplas ou nenhuma)
    moda_serie = serie.mode()
    moda_str = ", ".join([f"{x:.2f}" for x in moda_serie]) if not moda_serie.empty else "—"

    # Percentis adicionais úteis
    p25 = serie.quantile(0.25)
    p75 = serie.quantile(0.75)
    iqr = p75 - p25

    print(f"Variável: {var}")
    print(f"  Quantidade de observações válidas....: {n:>8}")
    print(f"  Menor valor..........................: {minimo:>12,.2f}")
    print(f"  Maior valor..........................: {maximo:>12,.2f}")
    print(f"  Amplitude............................: {amplitude:>12,.2f}")
    print(f"  Média................................: {media:>12,.2f}")
    print(f"  Mediana..............................: {mediana:>12,.2f}")
    print(f"  Moda.................................: {moda_str:>12}")
    print(f"  Variância............................: {variancia:>12,.2f}")
    print(f"  Desvio padrão........................: {desvio_padrao:>12,.2f}")
    print(f"  1º quartil (25%).....................: {p25:>12,.2f}")
    print(f"  3º quartil (75%).....................: {p75:>12,.2f}")
    print(f"  IQR (amplitude interquartil).........: {iqr:>12,.2f}")
    print("─" * 70)

# =============================================
# 3. Variáveis Qualitativas
# =============================================
print(" VARIÁVEIS QUALITATIVAS ".center(70, "═"))
print()

for var in qualitative_vars:
    if var not in df_arquivo.columns:
        continue

    contagem = df_arquivo[var].value_counts(dropna=False)
    total = len(df_arquivo)
    moda = df_arquivo[var].mode()
    moda_str = moda.iloc[0] if not moda.empty else "—"

    print(f"Variável: {var}")
    print(f"  Quantidade total de observações......: {total:>8}")
    print(f"  Moda.................................: {moda_str}")
    print("\nContagem por categoria:")
    print(contagem.to_string())
    print("─" * 70)
    print()

# =============================================
# 4. Resumo adicional interessante (opcional)
# =============================================
print(" RESUMO ADICIONAL ".center(70, "═"))

# Fibra = 1 por UF (proporção)
fibra_uf = df_arquivo.groupby('UF')['Fibra'].mean() * 100
print("\n% médio de municípios com Fibra = 1 por UF (ordenado):")
print(fibra_uf.sort_values(ascending=False).round(2))

# Municípios com maior PIB per capita
print("\nTop 5 municípios com maior PIB per capita:")
print(df_arquivo[['UF', 'nome_mun', 'PIB_per_capita', 'Populacao_2022']]
      .sort_values('PIB_per_capita', ascending=False)
      .head(5)
      .to_string(index=False))

print("\n" + "═" * 70)
print("Fim da análise descritiva")
print("═" * 70)



#=========================================================
# >>>>>> Porcentagem Presença de Fibra por UF <<<<<<<<<<<<
#=========================================================

# Criar tabela cruzada
tabela_fibra_uf = pd.crosstab(df_arquivo['UF'], df_arquivo['Fibra'],
                              margins=True,
                              margins_name='Total')


tabela_fibra_uf.columns = ['Fibra = 0', 'Fibra = 1', 'Total']

# Calcula proporção de Fibra = 1 (%)
tabela_fibra_uf['% Fibra = 1'] = (tabela_fibra_uf['Fibra = 1'] / tabela_fibra_uf['Total'] * 100).round(2)

# Ordenar por % Fibra = 1 descendente (opcional)
tabela_fibra_uf_ordenada = tabela_fibra_uf.sort_values('% Fibra = 1', ascending=False)
tabela_fibra_uf_ordenada.to_csv("UF_Fibra.csv", sep=';', encoding='utf-8')


#=========================================================#=================================
# >>>>>>>>>>>>>>>>>Proporção e contagem de Fibra = 1 e Fibra = 0 por PIB_UF <<<<<<<<<<<<<<<<
#=========================================================#=================================


df_IBC_PIB_POP = pd.read_csv("df_ibc_pib_pop.csv", sep=",", encoding="utf-8")

# Garantir tipos numéricos corretos
numeric_cols = ['PIB_per_capita', 'Populacao_2022']
df_PIB_POP = df_IBC_PIB_POP[['UF'] + numeric_cols].copy()
print(df_PIB_POP)

for col in numeric_cols:
    df_PIB_POP[col] = pd.to_numeric(df_PIB_POP[col], errors='coerce')

# Calcular PIB total de cada município
df_PIB_POP['PIB_mun'] = df_PIB_POP['PIB_per_capita'] * df_PIB_POP['Populacao_2022']
print(df_PIB_POP)
#
# Agrupar por UF e calcular as agregações
df_pib_uf = df_PIB_POP.groupby('UF', as_index=False).agg(
    Soma_PIB_mun=('PIB_mun', 'sum'),          # soma do PIB municipal
    Pop_Total=('Populacao_2022', 'sum')       # população total da UF
)

# 3. Calcular PIB per capita da UF (média ponderada)
df_pib_uf['PIB_UF'] = df_pib_uf['Soma_PIB_mun'] / df_pib_uf['Pop_Total']
#
df_percentFibra_PIB = pd.merge(df_pib_uf,
                               tabela_fibra_uf,
                               on='UF',
                               how='inner')

df_percentFibra_PIB = df_percentFibra_PIB[['UF', 'PIB_UF','% Fibra = 1']]
df_percentFibra_PIB.to_csv("UF_df_percentFibra_PIB.csv", sep=';', encoding='utf-8')

#=================================================================
# >>>>>>>>>>>>>>>>> REGRESSÃO LINEAR / CORRELAÇÃO <<<<<<<<<<<<<<<<
#=================================================================

dfFibraPIB = pd.read_csv("UF_df_percentFibra_PIB.csv", sep=";", decimal=".")

dfFibraPIB = dfFibraPIB.rename(columns={
    '% Fibra = 1': 'percent_fibra',
    'PIB_UF': 'pib_per_capita'
})

x = dfFibraPIB['percent_fibra']
y = dfFibraPIB['pib_per_capita']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(slope, intercept, r_value, p_value, std_err)
#

# ======================================================================
# >>>>>>>>>>>>>>>>  Gráfico de dispersão + linha de regressão <<<<<<<<<<
# ======================================================================

plt.figure(figsize=(10, 6))

# Scatter plot
sns.scatterplot(
    data=dfFibraPIB,
    x='percent_fibra',
    y='pib_per_capita',
    s=100,
    alpha=0.7,
    edgecolor='w'
)

# Linha de regressão linear (com intervalo de confiança)
sns.regplot(
    data=dfFibraPIB,
    x='percent_fibra',
    y='pib_per_capita',
    scatter=False,
    color='red',
    line_kws={'linewidth': 2},
    ci=95
)

# Adicionar rótulos das UFs (com pequeno offset para evitar sobreposição)
for i, uf in enumerate(dfFibraPIB['UF']):
    x_val = dfFibraPIB['percent_fibra'].iloc[i]
    y_val = dfFibraPIB['pib_per_capita'].iloc[i]
    plt.text(
        x_val + 0.4,
        y_val + 2000,  # pequeno ajuste vertical
        uf,
        fontsize=9,
        ha='center',
        alpha=0.85
    )

# Configurações do gráfico
plt.title('Relação entre % de municípios com Fibra Óptica e PIB per capita por UF\n'
          '(2022 – dados agregados)', fontsize=14, pad=15)
plt.xlabel('% de municípios com Fibra por UF = 1', fontsize=12)
plt.ylabel('PIB per capita da UF (R$)', fontsize=12)

# Formatar eixo y com separador de milhar
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.grid(True, alpha=0.3, linestyle='--')
sns.despine()
plt.tight_layout()
plt.show()



