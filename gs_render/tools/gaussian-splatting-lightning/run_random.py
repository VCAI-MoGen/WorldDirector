import os
import random
import subprocess
import shlex

# ① 루트 폴더 경로 지정
root_folder = "/home/ckd248/data/SceneSplat/dl3dv/FCGS/dataset/DL3DV-GS-960P/3DGS"  # 여기에 폴더 경로 입력
# root_folder = "/home/ckd248/data/SceneSplat/crowdsourced_3dgs_cleaned"

# ② 하위 폴더까지 모두 탐색해서 .ply 파일 경로 수집
ply_files = []
for dirpath, _, filenames in os.walk(root_folder):
    for f in filenames:
        if f.endswith('.ply'):
            ply_files.append(os.path.join(dirpath, f))

if not ply_files:
    raise FileNotFoundError("해당 폴더(및 하위 폴더)에 .ply 파일이 없습니다.")

# ③ 무작위로 하나 선택
chosen_file = random.choice(ply_files)

# ④ 절대 경로로 변환
abs_path = os.path.abspath(chosen_file)
abs_path = shlex.quote(abs_path)
print(f"선택된 파일: {abs_path}")

# ⑤ 실행할 터미널 명령어 설정
# 아래를 원하는 명령어로 바꾸세요
# 예: meshlab {abs_path} 또는 python your_script.py {abs_path}
command = f"python viewer.py {abs_path} --gsplat --root_path /home/ckd248/data/SCVideo/data/dl3dv_test/"

# ⑥ 명령어 실행
subprocess.run(command, shell=True)
