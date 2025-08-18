import os
import sys

with open('task.txt', 'r') as f:
    data = f.readlines()
data = [x.split('\\') for x in data]
print(data)
model = 'Wan2.1'

for image_path, prompt in data:
    if prompt[-1] == '\n':
        prompt = prompt[:-1]
    cmd = 'python run_i2v.py --image_path ' + image_path + ' --prompt ' + prompt + ' --model ' + model
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened. terminate the script.')
        sys.exit()