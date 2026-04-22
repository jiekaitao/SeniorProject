.PHONY: prebuild dev build down logs test test-backend test-frontend clean

# Copy TRM source files into worker/trm/ for Docker build
prebuild:
	@echo "Copying TRM source files..."
	rm -rf worker/trm
	mkdir -p worker/trm/models/recursive_reasoning worker/trm/evaluators worker/trm/dataset worker/trm/config/arch worker/trm/utils
	# Core training (modified for web)
	cp worker/pretrain_web.py worker/trm/pretrain_web.py
	cp worker/predict_web.py worker/trm/predict_web.py
	# Puzzle dataset
	cp ../RR_TRM/puzzle_dataset.py worker/trm/puzzle_dataset.py
	# Models (unmodified)
	cp ../RR_TRM/models/__init__.py worker/trm/models/__init__.py 2>/dev/null || touch worker/trm/models/__init__.py
	cp ../RR_TRM/models/common.py worker/trm/models/common.py
	cp ../RR_TRM/models/layers.py worker/trm/models/layers.py
	cp ../RR_TRM/models/sparse_embedding.py worker/trm/models/sparse_embedding.py
	cp ../RR_TRM/models/ema.py worker/trm/models/ema.py
	cp ../RR_TRM/models/losses.py worker/trm/models/losses.py
	cp ../RR_TRM/models/recursive_reasoning/__init__.py worker/trm/models/recursive_reasoning/__init__.py 2>/dev/null || touch worker/trm/models/recursive_reasoning/__init__.py
	cp ../RR_TRM/models/recursive_reasoning/trm.py worker/trm/models/recursive_reasoning/trm.py
	cp ../RR_TRM/models/recursive_reasoning/hrm.py worker/trm/models/recursive_reasoning/hrm.py 2>/dev/null || true
	cp ../RR_TRM/models/recursive_reasoning/trm_hier6.py worker/trm/models/recursive_reasoning/trm_hier6.py 2>/dev/null || true
	cp ../RR_TRM/models/recursive_reasoning/trm_singlez.py worker/trm/models/recursive_reasoning/trm_singlez.py 2>/dev/null || true
	cp ../RR_TRM/models/recursive_reasoning/transformers_baseline.py worker/trm/models/recursive_reasoning/transformers_baseline.py 2>/dev/null || true
	# CGAR models
	cp ../CGAR_TRM/models/recursive_reasoning/trm_cgar.py worker/trm/models/recursive_reasoning/trm_cgar.py 2>/dev/null || true
	cp ../CGAR_TRM/models/losses_cgar.py worker/trm/models/losses_cgar.py 2>/dev/null || true
	# Evaluators
	cp ../RR_TRM/evaluators/__init__.py worker/trm/evaluators/__init__.py 2>/dev/null || touch worker/trm/evaluators/__init__.py
	cp ../RR_TRM/evaluators/arc.py worker/trm/evaluators/arc.py 2>/dev/null || true
	# Dataset common
	cp ../RR_TRM/dataset/__init__.py worker/trm/dataset/__init__.py 2>/dev/null || touch worker/trm/dataset/__init__.py
	cp ../RR_TRM/dataset/common.py worker/trm/dataset/common.py
	# Utils
	cp ../RR_TRM/utils/__init__.py worker/trm/utils/__init__.py 2>/dev/null || touch worker/trm/utils/__init__.py
	cp ../RR_TRM/utils/functions.py worker/trm/utils/functions.py
	# Configs
	cp ../RR_TRM/config/cfg_pretrain.yaml worker/trm/config/cfg_pretrain.yaml
	cp ../RR_TRM/config/arch/trm.yaml worker/trm/config/arch/trm.yaml
	cp ../CGAR_TRM/config/arch/trm_cgar.yaml worker/trm/config/arch/trm_cgar.yaml 2>/dev/null || true
	@echo "TRM files copied successfully."

dev: prebuild
	docker compose up --build

build: prebuild
	docker compose build

down:
	docker compose down

logs:
	docker compose logs -f

test-backend:
	cd worker && python -m pytest __tests__/ -v --tb=short

test-frontend:
	cd frontend && npm test

test: test-backend test-frontend

clean:
	rm -rf worker/trm checkpoints uploads redis-data
	docker compose down -v
