activate g037

cd Z:\Source\Wizpresso Personal\Derek Yuen\mlp\project\Augmentation_Cropping

python ../model_main_tf2.py --pipeline_config_path=pipeline.config --model_dir=./ --alsologtostderr

cd Z:\Source\Wizpresso Personal\Derek Yuen\mlp\project\Augmentation_Gaussian

python ../model_main_tf2.py --pipeline_config_path=pipeline.config --model_dir=./ --alsologtostderr

cd Z:\Source\Wizpresso Personal\Derek Yuen\mlp\project\Augmentation_Normalisation

python ../model_main_tf2.py --pipeline_config_path=pipeline.config --model_dir=./ --alsologtostderr

echo done