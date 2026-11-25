.PHONY: install run dev test test-unit test-eval lint docker-up docker-down fmt

install:
	pip install -e ".[dev]"

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8080

dev:
	uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

test: test-unit test-eval

test-unit:
	pytest tests/unit -v

test-eval:
	pytest tests/evaluation -v

lint:
	ruff check src tests

fmt:
	ruff format src tests

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

logs:
	docker compose logs -f agent
