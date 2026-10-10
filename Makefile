.PHONY: up down smoke lint

up:
	docker compose up -d --wait

down:
	docker compose down

smoke:
	bash scripts/smoke.sh

lint:
	uvx --python 3.12 pre-commit@4.6.2 run --all-files
