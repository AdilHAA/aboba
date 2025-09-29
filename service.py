import os
import uuid
from typing import List

import pandas as pd
import torch
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ner_baseline import decode_predictions

app = FastAPI()

MODEL_DIR = os.environ.get('MODEL_DIR', 'models/ner')
PENDING = {}
_TOKENIZER = None
_MODEL = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PredictRequest(BaseModel):
	texts: List[str]


class SubmitRequest(BaseModel):
	input_csv: str
	output_csv: str


def _load_model():
	global _TOKENIZER, _MODEL
	if _TOKENIZER is None or _MODEL is None:
		_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
		_MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
		_MODEL.to(_DEVICE)
		_MODEL.eval()
	return _TOKENIZER, _MODEL


def _run_submit(job_id: str, input_csv: str, output_csv: str):
	try:
		tokenizer, model = _load_model()
		df = pd.read_csv(input_csv, sep=';', dtype=str).fillna('')
		rows = []
		for _, row in df.iterrows():
			text = row['sample']
			entities = decode_predictions(text, tokenizer, model)
			rows.append({'sample': text, 'annotation': str(entities)})
		pd.DataFrame(rows).to_csv(output_csv, sep=';', index=False)
		PENDING[job_id] = {'status': 'done', 'output_csv': output_csv}
	except Exception as e:
		PENDING[job_id] = {'status': 'error', 'error': str(e)}


@app.get('/health')
async def health():
	return {'status': 'ok'}


@app.post('/predict')
async def predict(req: PredictRequest):
	tokenizer, model = _load_model()
	res = []
	for text in req.texts:
		entities = decode_predictions(text, tokenizer, model)
		res.append(entities)
	return {'predictions': res}


@app.post('/submit')
async def submit(req: SubmitRequest, background: BackgroundTasks):
	job_id = str(uuid.uuid4())
	PENDING[job_id] = {'status': 'running'}
	background.add_task(_run_submit, job_id, req.input_csv, req.output_csv)
	return {'job_id': job_id}


@app.get('/submit/{job_id}')
async def submit_status(job_id: str):
	return PENDING.get(job_id, {'status': 'unknown'})
