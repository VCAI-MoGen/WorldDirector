import os
import subprocess

from core_pkg.common.utils.eval_utils import run_command_in_dir
from config import Config
cfg = Config()

result_dir = '/home/ckd248/data/SCVideo/camera_generation/tools/GenDoP/eval_results_director'
out, err, ret = run_command_in_dir(['python', '-m', 'src.extraction', f'checkpoint_path={cfg.clatr_ckpt_path}', f'data_dir={result_dir}'], \
                                    'evaluate/CLaTr')
print(out)
out, err, ret = run_command_in_dir(['python', '-m', 'src.eval_only', '--pred_path', f"{os.path.join(result_dir, 'preds.npy')}"], \
                                    'evaluate/eval')
print(out)