import torch
import sys, os, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../gaussian-splatting-lightning")))

SEND_PATH = "/data1/cympyc1785/SCVideo/gs_render/tools/infifo"
RECV_PATH = "/data1/cympyc1785/SCVideo/gs_render/tools/outfifo"
SAVE_PATH = "/data1/cympyc1785/SCVideo/gs_render/tools/cam_feat.pt"

def setup_pipe():
    if not os.path.exists(SEND_PATH):
        os.mkfifo(SEND_PATH)
    if not os.path.exists(RECV_PATH):
        os.mkfifo(RECV_PATH)

def get_optimization():
    with open(RECV_PATH, "r") as pipe:
        while True:
            time.sleep(1)
            path = pipe.readline().strip()
            if not path:
                continue
            try:
                print("Got :", path)
                tensor = torch.load(path)
                return tensor
            except Exception as e:
                print(f"Invalid PATH: {path}, error={e}")

def save_cam_feat(cam_feat: torch.Tensor):

    torch.save(cam_feat, SAVE_PATH)

    with open(SEND_PATH, "r") as pipe:
        pipe.write(SAVE_PATH + "\n")
        pipe.flush()

    print(f"Saved {SAVE_PATH}")

def get_scene_aware_gradient(cam_feat: torch.Tensor):
    """
    cam_feat : (B, P, T) - P : 9 = 6d + 3pos
    """
    setup_pipe()

    save_cam_feat(cam_feat)

    grad = get_optimization()

    return grad


if __name__ == "__main__":
    grad = get_scene_aware_gradient(torch.tensor([1, 2, 3]))
    print(grad)