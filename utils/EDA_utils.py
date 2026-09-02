"""
EDA_utils.py

Funcoes auxiliares de visualizacao e analise usadas no notebook EDA.ipynb.
Mantidas aqui separadas para deixar o notebook mais limpo e o codigo reutilizavel.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sns_plots(data, features, histplot=True, kde=True, boxplot=True, hue=None):
    for feature in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if histplot:
            sns.histplot(data=data, x=feature, hue=hue, kde=kde,
                         stat='density', common_norm=False, ax=axes[0])
            axes[0].set_title(f'Distribuição de {feature}')

        if boxplot:
            sns.boxplot(data=data, x=hue, y=feature, ax=axes[1])
            axes[1].set_title(f'Boxplot de {feature} por {hue}')

        plt.tight_layout()
        plt.show()


def contar_outliers(df, colunas):
    resultado = []

    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]

        qtd_outliers = len(outliers)
        percentual = (qtd_outliers / len(df)) * 100

        resultado.append({
            'coluna': col,
            'qtd_outliers': qtd_outliers,
            'percentual': round(percentual, 2)
        })

    return pd.DataFrame(resultado)


def plot_categoricas_bivariada(data, features, hue):
    for feature in features:
        # Calcula a proporção de cada categoria, separada por grupo do hue
        tabela = (data.groupby(hue)[feature]
                  .value_counts(normalize=True)
                  .rename('percentual')
                  .reset_index())

        plt.figure(figsize=(8, 4))
        sns.barplot(data=tabela, x=feature, y='percentual', hue=hue)
        plt.title(f'{feature} por {hue}')
        plt.xticks(rotation=45)
        plt.ylabel('Proporção dentro do grupo')
        plt.show()
