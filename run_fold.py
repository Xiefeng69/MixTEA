import os
import argparse
'''
    this file is aiming to automatically run the 5-fold cross validation:
    python run_fold.py
'''
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='all', choices=['all', 'zh_en', 'fr_en', 'ja_en', 'en_fr_15k', 'en_de_15k', 'd_w_15k', 'd_y_15k', 'en_fr_100k', 'en_de_100k', 'd_w_100k', 'd_y_100k'])
args = parser.parse_args()

if args.task == "all":
    task_list = ['en_fr_15k', 'en_de_15k', 'd_w_15k', 'd_y_15k']
else:
    task_list = [args.task]

for task in task_list:
    for fold in [1, 2, 3, 4, 5]:
        for il in ["sim_th"]: # ["sim_th", "bi_sim"]
            cmd = f"python run.py --task {task} --fold {fold} --il_method {il}"
            print(cmd)
            os.system(cmd)