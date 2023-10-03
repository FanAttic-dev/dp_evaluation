calibrate: 
	export PYTHONPATH="${PYTHONPATH}:./tvcalib" && \
	python calibrate.py

evaluate:
	export PYTHONPATH="${PYTHONPATH}:./tvcalib" && \
	python evaluate.py