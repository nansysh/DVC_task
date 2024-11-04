import pandas as pd


def preprocess_data(input_file, output_file):
    # Загрузка данных
    data = pd.read_csv(input_file)
    # Просмотр первых строк данных
    print("Исходные данные:")
    print(data.head())
    # Проверка на пропуски
    print("Проверка на пропуски:")
    print(data.isnull().sum())
    # Если есть пропуски, можно заполнить их (например, средним значением)
    # data.fillna(data.mean(), inplace=True)
    # Сохранение обработанных данных
    data.to_csv(output_file, index=False)
    print(f"Обработанные данные сохранены в {output_file}")


if __name__ == "__main__":
    input_file = 'data/raw/data.csv'  # Путь к исходным данным
    output_file = 'data/processed/data_processed.csv'  # Путь для сохранения обработанных данных
    preprocess_data(input_file, output_file)
