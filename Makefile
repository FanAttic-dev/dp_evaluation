export PYTHONPATH := "${PYTHONPATH}:./tvcalib"

calibrate: 
	echo ${PYTHONPATH}
	python calibrate.py

evaluate:
	python evaluate.py