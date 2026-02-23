import os
import json
import csv
import imageio.v2 as imageio
import cv2

def resize_video(input_path, output_path, target_size):
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    fps = meta['fps']
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1)
    for frame in reader:
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
        writer.append_data(resized)
    reader.close()
    writer.close()

def make_csv(video_path, text_data_path, output_csv):
    rows = []
    dataset_name_list = os.listdir(video_path)
    cnt = 0
    for dataset_name in dataset_name_list:
        dataset_path = os.path.join(video_path, dataset_name)
        data_name_list = os.listdir(dataset_path)
        
        for data_name in data_name_list:
            data_path = os.path.join(dataset_path, data_name)
            da3_path = os.path.join(data_path, 'da3')
            if os.path.exists(da3_path) is False:
                continue
            
            # images_path = os.path.join(data_path, 'images')
            # if len(os.listdir(images_path)) < 49:
            #     continue

            source_video_path = os.path.join(dataset_name, data_name, 'da3', 'inpainted', 'gs_video', '0000_original.mp4')
            if os.path.exists(os.path.join(video_path, source_video_path)) is False:
                continue

            reader = imageio.get_reader(os.path.join(video_path, source_video_path))
            src_num_frames = sum(1 for _ in reader)
            frame = reader.get_data(0)
            source_height, source_width = frame.shape[:2]
            reader.close()
            
            target_video_path = os.path.join(dataset_name, data_name, 'video_input_resized.mp4')
            if os.path.exists(os.path.join(video_path, target_video_path)) is False:
                original_video_path = os.path.join(dataset_name, data_name, 'video_input.mp4')
                if os.path.exists(os.path.join(video_path, original_video_path)) is False:
                    continue
                resize_video(os.path.join(video_path, original_video_path), os.path.join(video_path, target_video_path), (source_width, source_height))

            if os.path.exists(os.path.join(video_path, target_video_path)) is False or os.path.exists(os.path.join(video_path, source_video_path)) is False:
                continue

            reader = imageio.get_reader(os.path.join(video_path, target_video_path))
            target_num_frames = sum(1 for _ in reader)
            frame = reader.get_data(0)
            target_height, target_width = frame.shape[:2]
            reader.close()

            assert target_height == source_height and target_width == source_width

            if target_num_frames != src_num_frames:
                continue

            text_path = os.path.join(text_data_path, dataset_name, data_name, 'captions', 'object_caption.json')
            if os.path.exists(text_path) is False:
                continue
            with open(text_path, 'r') as f:
                text = json.load(f)
                if len(text) < 1:
                    continue
                if 'description' not in text[0]:
                    continue
                text = text[0]['description']

            rows.append({
                "video": f"{target_video_path}",
                "prompt": text,
                "src_video": f"{source_video_path}",
            })
            cnt += 1
            print(cnt)
            # if cnt > 1000:
            #     break

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "prompt", "src_video"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")

video_path = '/data1/cympyc1785/3d_recon/Depth-Anything-3/SceneData/DynamicVerse/scenes'
text_path = '/data1/cympyc1785/data/motion_dataset/DynamicVerse/data/DynamicVerseData'
make_csv(video_path, text_path, '../dataset/metadata_dv_0217.csv')
