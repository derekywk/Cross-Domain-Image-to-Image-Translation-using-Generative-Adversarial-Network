from object_detection import model_lib_v2
import os

def eval(start, end, step_size=1, directory='.', has_checkpoint=True, reverse=True):
    if has_checkpoint:
        os.rename(f'{directory}/checkpoint', f'{directory}/checkpoint_temp')
    steps = list(range(start, end+1, step_size))
    if end not in steps: steps.append(end)
    try:
        for step in steps:
            with open(f"{directory}/checkpoint", "w") as f:
                f.write(f'''model_checkpoint_path: "ckpt-{step}"
    all_model_checkpoint_paths: "ckpt-{step}"''')
            os.system(f"python model_main_tf2.py --pipeline_config_path={directory}/pipeline.config --model_dir={directory} --alsologtostderr --checkpoint_dir={directory} --num_train_steps={(step-1)*1000} --eval_timeout=10")
            os.remove(f"{directory}/checkpoint")
    except Exception as e:
        print("Error: ", directory, "ended at step", step)
        print(e)
        try:
            os.remove(f"{directory}/checkpoint")
        except: pass
    if has_checkpoint:
        os.rename(f'{directory}/checkpoint_temp', f'{directory}/checkpoint')


if __name__ == '__main__':
    try:
        eval(start=1, end=61, step_size=3, directory='./L2', has_checkpoint=True, reverse=False)
    except:
        pass
    try:
        eval(start=1, end=61, step_size=3, directory='./Augmentation_Cropping_with_L2', has_checkpoint=True, reverse=False)
    except:
        pass
    try:
        eval(start=1, end=61, step_size=3, directory='./Augmentation_All', has_checkpoint=True, reverse=False)
    except:
        pass