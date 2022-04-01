activate g037

cd .\Dropout_70

python ../model_main_tf2.py --pipeline_config_path=pipeline.config --model_dir=./ --alsologtostderr

echo done