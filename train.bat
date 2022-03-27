activate g037

cd Z:\Source\Wizpresso Personal\Derek Yuen\mlp\project\Augmentation_Cropping_with_L2_adjusted_LR

python ../model_main_tf2.py --pipeline_config_path=pipeline.config --model_dir=./ --alsologtostderr

cd Z:\Source\Wizpresso Personal\Derek Yuen\mlp\project\Augmentation_Cropping_with_L2

python ../model_main_tf2.py --pipeline_config_path=pipeline.config --model_dir=./ --alsologtostderr

echo done