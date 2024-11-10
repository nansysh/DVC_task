import json
import yaml
import joblib
import mlflow
import mlflow.sklearn  # Импортируем модуль для работы с моделями scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Загрузка гиперпараметров
with open('params.yaml') as f:
    params = yaml.safe_load(f)

data = pd.read_csv('data/processed/data_processed.csv')
X = data.drop('variety', axis=1)  # Предполагается, что целевая переменная называется 'target'
y = data['variety']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Начало нового эксперимента
with mlflow.start_run():
    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )

    model.fit(X_train, y_train)  # Обучаем модель на обучающей выборке

    # Предсказание и логирование метрик
    predictions = model.predict(X_test)  # Предсказание на тестовой выборке
    accuracy = accuracy_score(y_test, predictions)  # Вычисляем точность

    # Логирование метрик
    mlflow.log_metric("accuracy", accuracy)

    # Логирование гиперпараметров
    mlflow.log_params(params)

    # Сохранение модели
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Accuracy: {accuracy}")



# без MLflow
# # Обучение модели
# model = RandomForestClassifier(
#     n_estimators=params['n_estimators'],
#     max_depth=params['max_depth'],
#     random_state=42
# )
#
# model.fit(X, y)
# # Сохранение модели
# joblib.dump(model, 'models/model.pkl')  # Сохраняем модель в файл
# # Предсказание и логирование метрик
# predictions = model.predict(X)
# accuracy = accuracy_score(y, predictions)
#
# # Логирование метрик
# metrics = {'accuracy': accuracy}
# with open('metrics/metrics.json', 'w') as f:
#     json.dump(metrics, f)
#
# print(f"Accuracy: {accuracy}")

