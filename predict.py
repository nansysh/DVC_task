import mlflow
import pandas as pd

# Укажите путь к модели, зарегистрированной в MLflow
logged_model = 'runs:/a88598802ac34f209c888e346549a768/random_forest_model'

# Загрузка модели как PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Подготовка данных для предсказания
data = {
    "sepal.length": [5.2, 6.6],
    "sepal.width": [3.4, 3.2],
    "petal.length": [1.3, 5.5],
    "petal.width": [0.3, 2.2],
}

# Преобразование данных в DataFrame
input_data = pd.DataFrame(data)

# Выполнение предсказаний
predictions = loaded_model.predict(input_data)

# Вывод предсказаний
print("Predictions:", predictions)
