.PHONY: help install demo forecaster weather test lint api dashboard experiments clean

help:
	@echo "install      install dependencies"
	@echo "demo         run the full offline pipeline (RAG eval + landfall model)"
	@echo "forecaster   train + evaluate the landfall-intensity model"
	@echo "weather      build the Weather Intelligence landing page (HTML)"
	@echo "test         run unit + integration tests"
	@echo "lint         ruff lint"
	@echo "api          serve the FastAPI app on :8000"
	@echo "dashboard    launch the Streamlit console (Model Eval + Experiments)"
	@echo "experiments  run a chunk-size experiment sweep"
	@echo "clean        remove generated data artifacts"

install:
	pip install -r requirements.txt
	pip install -e .

demo:
	python scripts/run_local_pipeline.py

forecaster:
	python scripts/train_forecaster.py

weather:
	python scripts/build_weather_page.py
	@echo "open dashboards/weather_intelligence.html in a browser"

test:
	pytest -q

lint:
	ruff check src tests scripts

api:
	uvicorn storm_eval.serving.api:app --reload --port 8000

dashboard:
	streamlit run dashboards/Home.py

experiments:
	python scripts/run_experiment.py chunk_size 160 320 640

clean:
	rm -rf data/warehouse.duckdb data/vectorstore data/mlruns data/models
