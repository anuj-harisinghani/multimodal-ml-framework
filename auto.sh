conda activate canary2
python main.py
sed -i '6s/0/1/' ./params/settings.yaml
python main.py
sed -i '98s/1/2/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/2/3/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/3/4/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/4/5/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/5/6/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/6/7/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/7/8/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/8/9/' ./dementia_classifier/settings.py
python main.py
sed -i '98s/9/0/' ./dementia_classifier/settings.py


python create_result_tables.py ./csv_results/frontiers_review_with_SVM/TF_FINAL ./csv_results/frontiers_review_with_SVM/TF_FINAL/tf_final

python csv_to_table.py ./csv_results/frontiers_review_with_SVM/TF_FINAL/tf_final
