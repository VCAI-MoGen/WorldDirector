import os
import json
import csv

def make_csv(input_root, output_csv):
    rows = []
    json_path = os.path.join(input_root, 'add_data_configs.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    for idx, data_dict in enumerate(data):
        if idx > 1000:
            break
        src_video = data_dict['src_video'].replace("add/", "", 1)
        tar_video = data_dict['tar_video'].replace("add/", "", 1)
        text_prompt = data_dict['instruction_final_refine']

        assert os.path.exists(os.path.join(input_root, src_video)) and os.path.exists(os.path.join(input_root, tar_video))

        rows.append({
            "video": f"{tar_video}",
            "prompt": text_prompt,
            "src_video": f"{src_video}",
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "prompt", "src_video"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")

dataset_path = '/home/ckd248/data/SCVideo/camera_generation/tools/ReCo/ReCo-Data/add'
make_csv(dataset_path, '../dataset/metadata_ReCo_1K.csv')
