PYTHON=python3
PIP=pip3
MODEL?=default_model
setup:
	$(PIP) install -r requirements.txt
main:
	$(PYTHON) main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --model $(MODEL)
clean:
	rm -rf __pycache__ logs/*.log
help:
	@echo "Available commands:"
	@echo "make setup 	-Install dep"
	@echo "make main MODEL=name	-run the model"
	@echo "make clean	-clean temporary files"
