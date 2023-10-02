calibrate: export PYTHONPATH="${PYTHONPATH}:./tvcalib"
	python calibrate.py

evaluate:
	python evaluate.py