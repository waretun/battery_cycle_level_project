# Battery Cycle Level Project

## Запуск

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
c:/Users/waretun/Battery_Cycle_Level_project/.venv/Scripts/python.exe train.py --data battery_cycle_level_dataset_CLEAN_FINAL.csv --train-batteries B0005 B0006 B0007 --test-batteries B0018 --output results
```

## Настройки

- config: config.yml
- результаты: results_summary.txt
