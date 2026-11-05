.PHONY: up down logs ps user backfill sync test test-up lint web-dev api-dev

up:            ## build and start everything
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

user:          ## create the single login (password + TOTP QR code)
	docker compose run --rm backend python -m app.cli create-user

sync:          ## Kite instrument tokens + universe membership
	docker compose run --rm backend python -m app.cli sync-instruments

backfill:      ## load 5 years of daily history (resumable)
	docker compose run --rm backend python -m app.cli backfill

test-up:
	docker compose -f docker-compose.test.yml up -d

test: test-up  ## backend tests against throwaway Postgres/Redis
	cd backend && python -m pytest

lint:
	cd backend && ruff check app tests && ruff format --check app tests && mypy app
	cd frontend && npx tsc --noEmit
