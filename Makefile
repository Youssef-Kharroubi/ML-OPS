PYTHON=python3
PIP=pip3
MODEL?=default_model
setup:
	$(PIP) install -r requirements.txt
main:
	$(PYTHON) main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --model $(MODEL)
train:
	$(PYTHON) main.py train --train_data churn-bigml-80.csv --model $(MODEL) --save_model models/$(MODEL).pkl
test:
	$(PYTHON) main.py test --test_data churn-bigml-20.csv --model $(MODEL) --load_model models/$(MODEL).pkl
clean:
	rm -rf __pycache__ logs/*.log
help:
	@echo "Available commands:"
	@echo "make setup 				-Install dep"
	@echo "make train MODEL=name  	-train model"
	@echo "make test MODEL=name 	-test  model"
	@echo "make main MODEL=name		-run the model"
	@echo "make clean	-clean temporary files"
