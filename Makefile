calibrate: 
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python calibrate.py

evaluate:
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python evaluate.py

evaluate_show:
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python evaluate.py --show