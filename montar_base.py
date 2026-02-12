#############################################
# >>>>>>>>>>>>> BASE DE DADOS <<<<<<<<<<<<<<<
#############################################


import pandas as pd


# base IBC (CSV com ; e vírgula como decimal)
df_ibc = pd.read_csv(
    "IBC_municipios_indicadores_originais.csv",
    sep=";",
    decimal=",",
    encoding="utf-8",
    dtype={"Código Município": str}
    )

df_ibc = df_ibc.rename(columns={'Município':'nome_mun',"Código Município":"cod_mun" })
df_ibc["cod_mun"] = df_ibc["cod_mun"].str.zfill(7)
df_ibc = df_ibc[df_ibc["Ano"].isin([2022])]
# print(df_ibc)
# df_ibc.to_csv('df_ibc.csv', index=False)

# base PIB Municípios
df_pib = pd.read_excel(
    "PIB_mun.xlsx",
      sheet_name="PIB dos Municípios",
      dtype={"Ano": int,
             "Código do Município": str}

    )

df_pib = df_pib.rename(columns={'Sigla da Unidade da Federação':'UF',
                                'Nome do Município':'nome_mun',
                                'Produto Interno Bruto per capita, \na preços correntes\n(R$ 1,00)':'PIB_per_capita',
                                'Código do Município':'cod_mun'})


df_pib = df_pib[['Ano','UF', 'nome_mun', 'cod_mun', 'PIB_per_capita']]
df_pib["cod_mun"] = df_pib["cod_mun"].str.zfill(7)
df_pib = df_pib[df_pib["Ano"].isin([2022])]
# print(df_pib)
# df_pib.to_csv('df_pib.csv', index=False)


# base População Municípios/Área
df_populacao = pd.read_csv(
    "populacao_area_densidade_2022_todos_ufs.csv",
     sep=";",
     decimal=",",
     encoding="utf-8"
)

df_populacao = df_populacao.rename(columns={'Municipio':'nome_mun'})
df_populacao['Ano'] = 2022
# df_populacao.to_csv('df_populacao.csv', index=False)

# merge principal

df_ibc_pib = pd.merge(
    df_ibc,
    df_pib,
    how="inner",
    on=["cod_mun","Ano","UF"],
    suffixes=("_pib", "_ibc"),
)
df_ibc_pib['nome_mun'] = df_ibc_pib['nome_mun_ibc']

df_ibc_pib_novo = df_ibc_pib[['Ano', 'UF', 'IBC', 'Cobertura Pop. 4G5G',
       'Densidade SMP', 'HHI SMP', 'Densidade SCM', 'HHI SCM',
       'Adensamento Estações', 'Fibra',
       'PIB_per_capita', 'nome_mun']]
df_ibc_pib_novo.to_csv('df_ibc_pib_novo.csv', index=False)

# merge de df_ibc_pib_novo com df_pop

df_ibc_pib_pop = pd.merge(
    df_ibc_pib_novo,
    df_populacao,
    how="inner",
    on=["nome_mun","Ano","UF"]
)

# print(df_ibc_pib_pop)
df_ibc_pib_pop.to_csv('df_ibc_pib_pop.csv', index=False)




