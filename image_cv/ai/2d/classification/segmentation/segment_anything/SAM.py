# ======================================================= Library =======================================================
print("import Library")
# ----------------------- basic -----------------------
import argparse

# ----------------------- Data Science -----------------------
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- CV -----------------------
import cv2

# ----------------------- DL -----------------------
import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
CUDA_available = torch.cuda.is_available()
print("--CUDA is available:", CUDA_available, flush=True)


# ======================================================= import/dev func =======================================================
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 200/255, 0/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)   
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
 

if __name__ == "__main__":

    print("Main: Selecting objects with SAM Example")

    parser = argparse.ArgumentParser(
        description="""
        A script to run the object segmentation with SAM and different input types.
        This tool allows for precise control over the input type (points, bounding box) and the target image.
        
        Usage examples:
        1. Segment with points:
        python SAM.py --input_point '100,100;110,110;0,0' --input_label '1,1,0' --image_path './path/to/your/image.jpg'
        2. Segment with bounding box:
        python SAM.py --input_box '10,100,300,300' --image_path './path/to/your/image.jpg'
        3. Segment with points and bounding box:
        python SAM.py --input_point '100,100;110,110;0,0' --input_label '1,1,0' --input_box '10,100,300,300' --image_path './path/to/your/image.jpg'
        """
    )

    parser.add_argument("--input_point", "-ip", type=str, help="Input point coordinates as a string of comma-separated pairs, e.g., '100,100;110,110;0,0'", default=None)
    parser.add_argument("--input_label", "-il", type=str, help="Input labels for points as a comma-separated string, e.g., '1,1,0'", default=None)
    parser.add_argument("--input_box", "-ib", type=str, help="Input bounding box as a comma-separated string, e.g., '10,100,300,300'", default=None)
    parser.add_argument("--image_path", "-img_p", type=str, help="Path to the image file", default="./test_data/cat.jpg")

    args = parser.parse_args()

    # 轉換命令行參數
    input_point = np.array([list(map(int, pair.split(','))) for pair in args.input_point.split(';')]) if args.input_point else None
    input_label = np.array(list(map(int, args.input_label.split(',')))) if args.input_label else None
    input_box = np.array(list(map(int, args.input_box.split(',')))) if args.input_box else None
    image_path = args.image_path # "./test_data/cat.jpg"

    # input_point = None # np.array([[100, 100], [110, 110], [0, 0]])
    # input_label = None # np.array([1, 1, 0])
    # input_box = None # np.array([10, 100, 300, 300]) #這邊可以調整bounding box的位置和大小 h, w
    # image_path = args.image_path # "./test_data/cat.jpg"

    if input_box is not None:
        if input_box.shape == (4,):
            input_box_ = input_box[None, :]
        else:
            input_box_ = None
    else:
        input_box_ = None

    show = True
    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    # input_box = None

    # Detecting mode based on input availability
    if input_point is not None and input_label is not None and input_box is not None:
        mode = "points_and_boxes" # Combining points and boxes
    elif input_point is not None and input_label is not None:
        mode = "point" # Specifying a specific object with additional points
    elif input_box is not None:
        mode = "bounding_box" # Specifying a specific object with a box
    else:
        mode = "pass"
    print("--mode", mode, flush=True)

    if mode != "pass":
        image = cv2.imread(image_path) #放想要分割的圖片
        sam_checkpoint = "./model_checkpoint/sam_vit_h_4b8939.pth"  # 權重的路徑 download web_url: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        model_type = "vit_h"
        if CUDA_available:
            device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        if CUDA_available:
            sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords = input_point,
            point_labels = input_label,
            box = input_box_,
            multimask_output = True, # (the default setting) SAM outputs 3 masks. When False, it will return a single mask.
        ) # where scores gives the model's own estimation of the quality of these masks.
        if show:
            print('--masks.shape: ', masks.shape, flush=True)

        best_score = 0
        best_score_id = -1
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if best_score < score:
                best_score_id = i

        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(masks[best_score_id], plt.gca())
            if mode in ["point", "points_and_boxes"]:
                show_points(input_point, input_label, plt.gca())
            if mode in ["bounding_box", "points_and_boxes"]:
                show_box(input_box, plt.gca())

            plt.title(f"Mask {best_score_id}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
        print('Finish', flush=True)
    else:
        print('Pass', flush=True)
