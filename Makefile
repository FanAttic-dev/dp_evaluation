# export PYTHONPATH="$${PYTHONPATH}:./tvcalib"

calibrate: 
	python calibrate.py

evaluate:
	python evaluate.py