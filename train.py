import argparse
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def setup_logging(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['voltage_sq'] = data['voltage'] ** 2
    data['temp_sq'] = data['temperature'] ** 2
    data['cycle_voltage'] = data['cycle'] * data['voltage']
    return data


def validate_batteries(df: pd.DataFrame, batteries: list[str]) -> None:
    missing = set(batteries) - set(df['battery_id'].unique())
    if missing:
        raise ValueError(f"Батареи не найдены в датасете: {sorted(missing)}")


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError('Датасет пустой')
    return df


def train_and_evaluate(df: pd.DataFrame, train_ids: list[str], test_ids: list[str], output_dir: Path):
    validate_batteries(df, train_ids + test_ids)

    train = df[df['battery_id'].isin(train_ids)].copy()
    test = df[df['battery_id'].isin(test_ids)].copy()

    train = add_features(train)
    test = add_features(test)

    features = ['voltage', 'temperature', 'cycle', 'voltage_sq', 'temp_sq', 'cycle_voltage']

    X_train, y_train = train[features], train['soh']
    X_test, y_test = test[features], test['soh']

    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    results = {
        'mae_lr': mean_absolute_error(y_test, y_pred_lr),
        'rmse_lr': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae_rf': mean_absolute_error(y_test, y_pred_rf),
        'rmse_rf': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    }

    logging.info('Основной эксперимент: %s', results)

    # Эксперимент без cycle
    features_wo_cycle = ['voltage', 'temperature', 'voltage_sq', 'temp_sq']

    lr2 = LinearRegression()
    rf2 = RandomForestRegressor(random_state=42)
    lr2.fit(X_train[features_wo_cycle], y_train)
    rf2.fit(X_train[features_wo_cycle], y_train)

    y_pred_lr2 = lr2.predict(X_test[features_wo_cycle])
    y_pred_rf2 = rf2.predict(X_test[features_wo_cycle])

    results['mae_lr_wo_cycle'] = mean_absolute_error(y_test, y_pred_lr2)
    results['mae_rf_wo_cycle'] = mean_absolute_error(y_test, y_pred_rf2)

    logging.info('Эксперимент без cycle: %s', {
        'mae_lr': results['mae_lr_wo_cycle'],
        'mae_rf': results['mae_rf_wo_cycle']
    })

    # Анализ важности признаков
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    logging.info('Важность признаков:\n%s', importance.to_string(index=False, float_format='%.4f'))

    # Ошибки по зонам SOH (LR)
    df_test = test.copy()
    df_test['pred_lr'] = y_pred_lr

    zones = {
        'high': df_test[df_test['soh'] >= 0.9],
        'mid': df_test[(df_test['soh'] >= 0.8) & (df_test['soh'] < 0.9)],
        'low': df_test[df_test['soh'] < 0.8],
    }

    for zone_name, zone_df in zones.items():
        if len(zone_df) > 0:
            results[f'mae_lr_zone_{zone_name}'] = mean_absolute_error(zone_df['soh'], zone_df['pred_lr'])
        else:
            results[f'mae_lr_zone_{zone_name}'] = np.nan

    # Сохранение моделей
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(lr, model_dir / 'linear_regression.pkl')
    joblib.dump(rf, model_dir / 'random_forest.pkl')

    # Визуализации
    output_dir.mkdir(exist_ok=True, parents=True)

    def save_plot(fig, filename):
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logging.info('Сохранен график %s', path)

    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(test['cycle'], y_test, label='Реальный SOH', linewidth=2, color='black')
    plt.plot(test['cycle'], y_pred_lr, label='Линейная регрессия', linestyle='--', alpha=0.8)
    plt.plot(test['cycle'], y_pred_rf, label='Случайный лес', linestyle='-.', alpha=0.8)
    plt.xlabel('Номер цикла')
    plt.ylabel('SOH (%)')
    plt.title('Деградация аккумулятора (тестовая батарея)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(fig1, 'degradation_plot.png')

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred_lr, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[0].set_xlabel('Реальный SOH')
    axes[0].set_ylabel('Предсказанный SOH')
    axes[0].set_title(f'Линейная регрессия\nMAE = {results["mae_lr"]:.4f}, RMSE = {results["rmse_lr"]:.4f}')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, y_pred_rf, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[1].set_xlabel('Реальный SOH')
    axes[1].set_ylabel('Предсказанный SOH')
    axes[1].set_title(f'Случайный лес\nMAE = {results["mae_rf"]:.4f}, RMSE = {results["rmse_rf"]:.4f}')
    axes[1].grid(alpha=0.3)

    fig2.tight_layout()
    save_plot(fig2, 'scatter_plot.png')

    df_test['error'] = np.abs(df_test['soh'] - df_test['pred_lr'])
    fig3 = plt.figure(figsize=(12, 5))
    plt.plot(df_test['cycle'], df_test['error'], color='red', linewidth=1.5)
    plt.xlabel('Номер цикла')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Ошибка модели (линейная регрессия) по циклам')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(fig3, 'error_plot.png')

    fig4 = plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'], color='steelblue')
    plt.xlabel('Важность')
    plt.title('Важность признаков (Случайный лес)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_plot(fig4, 'feature_importance.png')

    return results


def main():
    parser = argparse.ArgumentParser(description='Battery SOH regression pipeline')
    parser.add_argument('--data', default='battery_cycle_level_dataset_CLEAN_FINAL.csv', help='CSV файл с данными')
    parser.add_argument('--train-batteries', nargs='+', default=['B0005', 'B0006', 'B0007'], help='Список батарей для обучения')
    parser.add_argument('--test-batteries', nargs='+', default=['B0018'], help='Список батарей для теста')
    parser.add_argument('--output', default='results', help='Папка для результатов')
    parser.add_argument('--debug', action='store_true', help='Включить подробный лог')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    log_file = output_dir / 'training.log'
    setup_logging(log_file)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Запуск анализа аккумуляторных данных')

    df = load_data(Path(args.data))
    logging.info('Данные загружены. Форма: %s', df.shape)
    logging.info('Доступные батареи: %s', sorted(df['battery_id'].unique()))

    results = train_and_evaluate(df, args.train_batteries, args.test_batteries, output_dir)

    summary_file = output_dir / 'results_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        for k, v in results.items():
            f.write(f'{k}: {v}\n')

    # Сохраняем конфигурацию запуска в YAML
    config_file = output_dir / 'config.yml'
    config_str = (
        f"data: {args.data}\n"
        f"train_batteries: {args.train_batteries}\n"
        f"test_batteries: {args.test_batteries}\n"
        f"output: {args.output}\n"
        f"debug: {args.debug}\n"
    )
    config_file.write_text(config_str, encoding='utf-8')
    logging.info('Сохранена конфигурация %s', config_file)

    logging.info('Эксперимент завершён. Сводка: %s', results)
    logging.info('Сохранено в %s', summary_file)

    # Создаём README с командой запуска
    readme_file = Path('README.md')
    readme_text = (
        '# Battery Cycle Level Project\n\n'
        '## Запуск\n\n'
        '```powershell\n'
        'Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned\n'
        '& .\\.venv\\Scripts\\Activate.ps1\n'
        'c:/Users/waretun/Battery_Cycle_Level_project/.venv/Scripts/python.exe train.py --data battery_cycle_level_dataset_CLEAN_FINAL.csv --train-batteries B0005 B0006 B0007 --test-batteries B0018 --output results\n'
        '```\n\n'
        '## Настройки\n\n'
        f'- config: {config_file.name}\n'
        f'- результаты: {summary_file.name}\n'
    )
    readme_file.write_text(readme_text, encoding='utf-8')
    logging.info('Сохранён README %s', readme_file)


if __name__ == '__main__':
    main()
