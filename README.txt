Запуск тренировки:
python ner_baseline.py --train --data dataset/Датасет/train.csv --out models/ner

Инференс файла в формат сабмита:
python ner_baseline.py --infer --input dataset/Датасет/train.csv --output submission.csv

Сервис:
uvicorn service:app --host 0.0.0.0 --port 8000

Эндпоинты:
GET /health
POST /predict {"texts": ["абрикосы 500г global village"]}
POST /submit {"input_csv": "dataset/Датасет/train.csv", "output_csv": "submission.csv"}
GET /submit/{job_id}
