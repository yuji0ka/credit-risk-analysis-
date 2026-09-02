"""
modelagem_utils.py

Funcoes auxiliares de avaliacao e interpretacao de modelos usadas no
notebook modelagem_v5.ipynb. Mantidas aqui separadas para deixar o
notebook mais limpo e o codigo reutilizavel.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_models_cv(models, X_train, y_train, preprocessor):
  n_folds = 5
  stratified_kfold = StratifiedKFold(
      n_splits=n_folds, shuffle=True, random_state=42
  )

  models_val_scores = {}
  models_train_scores = {}

  for model_name, model in models.items():
    pipeline_completo = Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', model)]
    )

    # 1. Validação Cruzada (Métrica real)
    val_scores = cross_val_score(
        pipeline_completo,
        X_train,
        y_train,
        scoring='roc_auc',
        cv=stratified_kfold,
    )
    avg_val_score = val_scores.mean()
    val_score_std = val_scores.std()

    # 2. Fit único no treino total para diagnóstico de overfitting e tempo
    start_time = time.time()
    pipeline_completo.fit(X_train, y_train)
    training_time = time.time() - start_time

    if hasattr(pipeline_completo.named_steps['classifier'], 'predict_proba'):
      y_train_score = pipeline_completo.predict_proba(X_train)[:, 1]
    else:
      y_train_score = pipeline_completo.decision_function(X_train)

    train_score = roc_auc_score(y_train, y_train_score)

    models_val_scores[model_name] = avg_val_score
    models_train_scores[model_name] = train_score

    print(f'{model_name} Resultados:')
    print('-' * 50)
    print(f'Score de Treino (AUC): {train_score:.4f}')
    print(
        f'Score Médio de Validação (AUC): {avg_val_score:.4f} (+/-'
        f' {val_score_std:.4f})'
    )
    print(f'Tempo de Ajuste: {training_time:.4f} segundos\n')

  eval_df = pd.DataFrame({
      'Model': list(models_val_scores.keys()),
      'Average Val Score': list(models_val_scores.values()),
      'Train Score': list(models_train_scores.values()),
  })

  return eval_df


def importancia_por_variavel_original(pipeline):
    """
    Calcula a importancia de cada VARIAVEL ORIGINAL (antes do encoding),
    somando a importancia de todas as colunas que o OneHotEncoder gerou
    a partir dela. Sem isso, uma variavel como 'checking_account' apareceria
    fatiada em 3-4 barras separadas (uma por categoria) em vez de uma so.
    """
    modelo = pipeline.named_steps['classifier']
    preprocessador = pipeline.named_steps['preprocessor']
    importancias = modelo.feature_importances_

    variavel_de_cada_coluna = []
    for _, transformador, colunas_entrada in preprocessador.transformers_:
        if transformador in ('drop', 'passthrough'):
            continue
        nomes_saida = transformador.get_feature_names_out(colunas_entrada)
        for nome_saida in nomes_saida:
            # Uma coluna gerada pelo OneHotEncoder tem o nome
            # 'coluna_original_categoria' (ex: 'checking_account_A12').
            # Aqui a gente descobre a qual coluna original ela pertence.
            candidatos = [c for c in colunas_entrada if nome_saida == c or nome_saida.startswith(c + '_')]
            variavel_de_cada_coluna.append(candidatos[0] if candidatos else nome_saida)

    importancia_df = pd.DataFrame({
        'variavel': variavel_de_cada_coluna,
        'importancia': importancias
    })

    return importancia_df.groupby('variavel')['importancia'].sum().sort_values(ascending=False)


def plotar_importancia_variaveis(pipeline, top_n=None):
    importancias_agrupadas = importancia_por_variavel_original(pipeline)

    if top_n is not None:
        importancias_agrupadas = importancias_agrupadas.head(top_n)

    plt.figure(figsize=(12, 5))
    plt.bar(
        importancias_agrupadas.index,
        importancias_agrupadas.values,
        color=plt.cm.tab20.colors[:len(importancias_agrupadas)]
    )
    plt.title('Importância das Variáveis (Random Forest)')
    plt.ylabel('Importância (redução média de impureza)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return importancias_agrupadas
