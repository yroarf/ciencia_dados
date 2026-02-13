import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

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
print(quantitative_vars)

# =============================================
#  Variáveis Quantitativas
# =============================================
print("-------- Variáveis Quantitativas-------")
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

# =============================================
#  Variáveis Qualitativas
# =============================================

print("-------- Variáveis Qualitativas")
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

# =============================================
# Resumo adicional interessante (opcional)
# =============================================
print(" RESUMO ADICIONAL ".center(70, "═"))

# Fibra = 1 por UF (proporção)
fibra_uf = df_arquivo.groupby('UF')['Fibra'].mean() * 100
print("\n% médio de municípios com Fibra = 1 por UF (ordenado):")
print(fibra_uf.sort_values(ascending=False).round(2))


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
# print(df_PIB_POP)

for col in numeric_cols:
    df_PIB_POP[col] = pd.to_numeric(df_PIB_POP[col], errors='coerce')

# Calcular PIB total de cada município
df_PIB_POP['PIB_mun'] = df_PIB_POP['PIB_per_capita'] * df_PIB_POP['Populacao_2022']
# print(df_PIB_POP)
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
plt.xlabel('% de municípios com Fibra = 1', fontsize=12)
plt.ylabel('PIB per capita da UF (R$)', fontsize=12)

# Formatar eixo y com separador de milhar
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.grid(True, alpha=0.3, linestyle='--')
sns.despine()
plt.tight_layout()
plt.show()


#======================================================================
# >>>>>>>>>>>>>>>>>>> CORRELAÇÃO MULTIVARIÁVEL IBC, %FIBRA E PIB <<<<<<<
#=======================================================================

# =============================================
# Leitura e preparação
# =============================================
df = pd.read_csv("df_ibc_pib_pop.csv", sep=",", encoding="utf-8")

# Converter para numérico
for col in ['PIB_per_capita', 'IBC', 'Fibra', 'Populacao_2022']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover linhas inválidas
df = df.dropna(subset=['UF', 'PIB_per_capita', 'IBC', 'Fibra', 'Populacao_2022'])

print(f"Registros válidos: {len(df):,}\n")

# =============================================
# Agregação por UF
# =============================================
uf_agg = df.groupby('UF').agg(
    pop_total      = ('Populacao_2022', 'sum'),
    pib_total      = ('PIB_per_capita', lambda x: np.average(x, weights=df.loc[x.index, 'Populacao_2022'])),
    ibc_medio      = ('IBC', 'mean'),
    fibra_medio    = ('Fibra', 'mean'),
    n_municipios   = ('nome_mun', 'count')
).reset_index()

uf_agg['fibra_pct'] = uf_agg['fibra_medio'] * 100

print("Dados agregados por UF (ordenado por PIB per capita):")
print(uf_agg[['UF', 'pib_total', 'ibc_medio', 'fibra_pct', 'n_municipios']]
      .rename(columns={'pib_total': 'PIB_per_capita', 'ibc_medio': 'IBC_médio', 'fibra_pct': '% Fibra'})
      .round(2)
      .sort_values('PIB_per_capita', ascending=False))
uf_agg.to_csv('uf_agg.csv', index=False)
# =============================================
# Correlações
# =============================================

correl = uf_agg[['pib_total', 'ibc_medio', 'fibra_medio']].corr().round(3)

print("\nMatriz de correlação (nível UF agregado):")
print(correl)

print("\nCorrelação com PIB per capita:")
print(correl['pib_total'].sort_values(ascending=False))

# =============================================
# Regressão linear múltipla
# =============================================
X = uf_agg[['ibc_medio', 'fibra_medio']]
y = uf_agg['pib_total']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()


# =============================================
# Gráfico comparativo
# =============================================
plt.figure(figsize=(10, 6))

sns.regplot(x=uf_agg['ibc_medio'], y=uf_agg['pib_total'],
            scatter_kws={'s':80, 'alpha':0.7}, line_kws={'color':'darkblue'},
            label=f'IBC médio (r = {uf_agg["pib_total"].corr(uf_agg["ibc_medio"]):.3f})')

sns.regplot(x=uf_agg['fibra_pct'], y=uf_agg['pib_total'],
            scatter_kws={'s':80, 'alpha':0.7}, line_kws={'color':'darkred', 'linestyle':'--'},
            label=f'% Fibra (r = {uf_agg["pib_total"].corr(uf_agg["fibra_medio"]):.3f})')

for i, uf in enumerate(uf_agg['UF']):
    plt.text(uf_agg['ibc_medio'].iloc[i]+0.2, uf_agg['pib_total'].iloc[i]+500, uf, fontsize=8.5)

plt.title('PIB per capita × IBC médio e % Fibra (por UF)')
plt.xlabel('Valor da variável explicativa')
plt.ylabel('PIB per capita médio ponderado (R$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#######################################################################################
#################### ANÁLISE - OUTLIERS    ###########################################
#######################################################################################

# =============================================
# Leitura e preparação
# =============================================
df = pd.read_csv("df_ibc_pib_pop.csv", sep=",", encoding="utf-8")

# Converter para numérico
quant_vars = ['PIB_per_capita', 'IBC', 'Populacao_2022', 'Densidade_hab_km2', 'Fibra']
for col in quant_vars:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Variáveis qualitativas escolhidas
qual_vars = ['UF', 'Fibra', 'Ano']

print(f"Total de linhas originais: {len(df):,}\n")


# =============================================
# Função para calcular estatísticas e remover outliers por grupo
# =============================================
def remove_outliers(df, quant_col, group_col=None, multiplier=1.5):

    df_outliers = df.copy()

    if group_col:
        groups = df_outliers[group_col].unique()
        resultados = []

        for g in groups:
            subset = df_outliers[df_outliers[group_col] == g].copy()
            if len(subset) < 4:  # poucos dados → não faz sentido remover
                continue

            Q1 = subset[quant_col].quantile(0.25)
            Q3 = subset[quant_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR

            media_antes = subset[quant_col].mean()
            mediana_antes = subset[quant_col].median()
            n_antes = len(subset)

            # Remove outliers
            subset_novo = subset[(subset[quant_col] >= lower) & (subset[quant_col] <= upper)]
            media_depois = subset_novo[quant_col].mean()
            mediana_depois = subset_novo[quant_col].median()
            n_depois = len(subset_novo)
            outliers_removidos = n_antes - n_depois

            resultados.append({
                'grupo': g,
                'n_original': n_antes,
                'n_apos_remocao': n_depois,
                'outliers_removidos': outliers_removidos,
                'media_antes': round(media_antes, 2),
                'mediana_antes': round(mediana_antes, 2),
                'media_depois': round(media_depois, 2) if n_depois > 0 else np.nan,
                'mediana_depois': round(mediana_depois, 2) if n_depois > 0 else np.nan
            })

        return pd.DataFrame(resultados)

    else:
        # Sem agrupamento
        Q1 = df_outliers[quant_col].quantile(0.25)
        Q3 = df_outliers[quant_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        media_antes = df_outliers[quant_col].mean()
        mediana_antes = df_outliers[quant_col].median()
        n_antes = len(df_outliers)

        df_novo = df_outliers[(df_outliers[quant_col] >= lower) & (df_outliers[quant_col] <= upper)]
        media_depois = df_novo[quant_col].mean()
        mediana_depois = df_novo[quant_col].median()
        n_depois = len(df_novo)

        return {
            'variavel': quant_col,
            'n_original': n_antes,
            'n_apos_remocao': n_depois,
            'outliers_removidos': n_antes - n_depois,
            'media_antes': round(media_antes, 2),
            'mediana_antes': round(mediana_antes, 2),
            'media_depois': round(media_depois, 2),
            'mediana_depois': round(mediana_antes, 2)
        }


# =============================================
# 3. Análise por agrupamento (UF e Fibra)
# =============================================
print("=== Análise de outliers por UF ===")
for var in ['PIB_per_capita', 'IBC', 'Populacao_2022']:
    print(f"\nVariável: {var}")
    result_uf = remove_outliers(df, var, group_col='UF')
    print(result_uf)

print("\n=== Análise de outliers por Fibra (0 vs 1) ===")
for var in ['PIB_per_capita', 'IBC', 'Populacao_2022']:
    print(f"\nVariável: {var}")
    result_fibra = remove_outliers(df, var, group_col='Fibra')
    print(result_fibra)

# =============================================
#    Comparação global (sem agrupamento)
# =============================================
print("\n=== Comparação GLOBAL (sem agrupar) ===")
for var in ['PIB_per_capita', 'IBC', 'Populacao_2022', 'Densidade_hab_km2']:
    resultado = remove_outliers(df, var)
    print(f"\n{var}:")
    print(f"Antes → média: {resultado['media_antes']:,} | mediana: {resultado['mediana_antes']:,}")
    print(f"Depois → média: {resultado['media_depois']:,} | mediana: {resultado['mediana_depois']:,}")
    print(
        f"Outliers removidos: {resultado['outliers_removidos']} ({resultado['outliers_removidos'] / resultado['n_original'] * 100:.1f}%)")


# =============================================
#    Leitura e preparação
# =============================================
df = pd.read_csv("df_ibc_pib_pop.csv", sep=",", encoding="utf-8")

# Variáveis que vamos analisar
variaveis = ['PIB_per_capita', 'IBC', 'Populacao_2022', 'Densidade_hab_km2']

# Converter para numérico
for col in variaveis:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover NaN para essas colunas
df = df.dropna(subset=variaveis).copy()

print(f"Linhas válidas para análise: {len(df):,}\n")


# =============================================
# Função para remover outliers (IQR global)
# =============================================
def remove_outliers_global(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    df_clean = df[(df[column] >= lower) & (df[column] <= upper)].copy()

    stats_before = {
        'n': len(df),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'min': df[column].min(),
        'max': df[column].max()
    }

    stats_after = {
        'n': len(df_clean),
        'mean': df_clean[column].mean(),
        'median': df_clean[column].median(),
        'min': df_clean[column].min(),
        'max': df_clean[column].max()
    }

    return df_clean, stats_before, stats_after


# =============================================
#  Preparar dados para todos os gráficos
# =============================================

resultados = {}
dfs_clean = {}

for var in variaveis:
    df_clean, stats_b, stats_a = remove_outliers_global(df, var)
    dfs_clean[var] = df_clean
    resultados[var] = {'antes': stats_b, 'depois': stats_a}

# Mostrar tabela comparativa
print("Comparação estatística antes × depois da remoção de outliers\n")
comparacao = []
for var in variaveis:
    b = resultados[var]['antes']
    a = resultados[var]['depois']
    comparacao.append({
        'Variável': var,
        'N antes': b['n'],
        'N depois': a['n'],
        'Média antes': round(b['mean'], 2),
        'Média depois': round(a['mean'], 2),
        'Mediana antes': round(b['median'], 2),
        'Mediana depois': round(a['median'], 2),
        'Outliers removidos': b['n'] - a['n']
    })

df_comp = pd.DataFrame(comparacao)
print(df_comp.to_string(index=False))
df_comp.to_csv('df_comp.csv', index=False)

# =============================================
#  Gerar os gráficos
# =============================================
fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharey=False)
fig.suptitle("Comparação Antes × Depois da Remoção de Outliers (método IQR 1.5)",
             fontsize=18, fontweight='bold')

for i, var in enumerate(variaveis):
    # Boxplot antes - mais estreito
    sns.boxplot(y=df[var], ax=axes[i, 0], color='lightblue', width=0.25)
    axes[i, 0].set_title(f"{var} – Antes", fontsize=14)
    axes[i, 0].set_ylabel(var, fontsize=12)
    axes[i, 0].tick_params(axis='both', labelsize=11)
    axes[i, 0].grid(True, alpha=0.3, axis='y')

    # Boxplot depois - mais estreito
    sns.boxplot(y=dfs_clean[var][var], ax=axes[i, 1], color='salmon', width=0.25)
    axes[i, 1].set_title(f"{var} – Depois", fontsize=14)
    axes[i, 1].set_ylabel("")
    axes[i, 1].tick_params(axis='both', labelsize=11)
    axes[i, 1].grid(True, alpha=0.3, axis='y')

    # Adicionar texto com estatísticas (fonte maior)
    b = resultados[var]['antes']
    a = resultados[var]['depois']
    texto = f"Antes:\nMédia: {b['mean']:,.2f}\nMediana: {b['median']:,.2f}\nN: {b['n']:,}\n\nDepois:\nMédia: {a['mean']:,.2f}\nMediana: {a['median']:,.2f}\nN: {a['n']:,}"
    axes[i, 1].text(1.08, 0.5, texto, transform=axes[i, 1].transAxes,
                    va='center', ha='left', fontsize=11.5,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#######################################################################################
################   CRUZAMENTO DE VARIÁVEIS  ########################################
#######################################################################################

# =============================================
#  Leitura e preparação dos dados
# =============================================

df_cruz = pd.read_csv("df_ibc_pib_pop.csv", sep=",", decimal=".", encoding="utf-8")

# Renomear algumas colunas para facilitar (caso necessário)
df_cruz = df_cruz.rename(columns={
    'PIB_per_capita': 'pib_pc',
    'HHI SMP': 'hhi_smp',
    'Cobertura Pop. 4G5G': 'cob_4g5g',
    'Densidade SMP': 'dens_smp'
})

# Garantir que as colunas numéricas estejam no formato correto
for col in ['IBC', 'hhi_smp', 'pib_pc', 'Populacao_2022', 'Area_km2']:
    df_cruz[col] = pd.to_numeric(df_cruz[col], errors='coerce')

#Remover linhas com valores faltantes nas variáveis principais (opcional, mas ajuda na análise)
df_limpa = df_cruz.dropna(subset=['UF', 'IBC', 'hhi_smp', 'pib_pc']).copy()

print("Shape após limpeza:", df_limpa.shape)
print("Estados presentes:", sorted(df_limpa['UF'].unique()))

# =============================================
#  Estatísticas por UF
# =============================================

stats = df_limpa.groupby('UF').agg({
    'IBC': ['mean', 'std'],
    'hhi_smp': ['mean', 'std'],
    'pib_pc': ['mean', 'std', 'count']
}).round(1)

# Ajustar nomes das colunas
stats.columns = [
    'IBC_mean', 'IBC_std',
    'HHI_mean', 'HHI_std',
    'PIB_mean', 'PIB_std', 'n_municipios'
]

# Ordenar por IBC descendente
stats = stats.sort_values('IBC_mean', ascending=False)

print("\n=== Estatísticas por UF (ordenado por média de IBC) ===\n")
print(stats)
df_estatistica = pd.DataFrame(stats)
df_estatistica.to_csv("df_estatistica.csv", sep= ";" ,index=False)




