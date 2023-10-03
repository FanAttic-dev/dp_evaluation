base:
	export PYTHONPATH="$${PYTHONPATH}:./tvcalib"

calibrate: base
	python calibrate.py

evaluate: base
	python evaluate.py

evaluate_show: base
	python evaluate.py --show