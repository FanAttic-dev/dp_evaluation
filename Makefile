all: export_var_frames calibrate evaluate

export_var_frames:
	python src/export_var_frames.py

calibrate: 
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python src/calibrate.py

evaluate:
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python src/evaluate.py

evaluate_show:
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python src/evaluate.py --show

evaluate_fig_save:
	export PYTHONPATH="${PYTHONPATH}:./src/tvcalib" && \
	python src/evaluate.py --fig_save
