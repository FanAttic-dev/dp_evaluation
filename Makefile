calibrate: 
	export PYTHONPATH="${PYTHONPATH}:./tvcalib" && \
	python calibrate.py

evaluate:
	export PYTHONPATH="${PYTHONPATH}:./tvcalib" && \
	python evaluate.py

evaluate_show:
	export PYTHONPATH="${PYTHONPATH}:./tvcalib" && \
	python evaluate.py --show