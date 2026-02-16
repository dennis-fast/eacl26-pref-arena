PYTHON ?= python
PORT ?= 8000

.PHONY: help sync build-projection build-all serve-pages serve-flask smoke-flask test

help:
	@echo "Targets:"
	@echo "  make sync              - sync src/web -> docs + legacy/web"
	@echo "  make build-projection  - rebuild docs/data/projection_pca.json"
	@echo "  make build-all         - sync + build-projection"
	@echo "  make serve-pages       - local static preview at :$(PORT)"
	@echo "  make serve-flask       - run Flask legacy backend"
	@echo "  make smoke-flask       - API smoke test via test client"
	@echo "  make test              - run smoke tests"

sync:
	$(PYTHON) scripts/build/sync_web_assets.py

build-projection:
	$(PYTHON) scripts/build/build_static_projection.py

build-all: sync build-projection

serve-pages:
	$(PYTHON) -m http.server $(PORT)

serve-flask:
	$(PYTHON) server.py

smoke-flask:
	$(PYTHON) -c "from server import app; c=app.test_client(); print('root', c.get('/').status_code); print('meta', c.get('/api/meta').status_code); print('topic', c.post('/api/topic_match', json={'ratings': {}, 'slots': []}).status_code)"

test:
	$(PYTHON) -m unittest tests/test_smoke.py -v
