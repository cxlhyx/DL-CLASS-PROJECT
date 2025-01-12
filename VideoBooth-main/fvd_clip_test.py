import os
import cv2
import torch
import os
import subprocess
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pytorch_fid
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score


# Function to calculate FID
# def calculate_fid(real_path, generated_path, device):
#     try:
#         inception = torchvision.models.inception_v3(weights="DEFAULT", transform_input=True)
#
#         fid_value = fid_score.calculate_fid_given_paths([real_path, generated_path],
#                                                         inception,
#                                                         dims=2048,
#
#                                                         device=device)
#         print('FID value:', fid_value)
#
#     except Exception as e:
#         print(f"Error calculating FID: {e}")


# Function to calculate CLIP Score
def calculate_clip_score(image_path, text_prompt):
    image = Image.open(image_path)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像->文本的分数

    return logits_per_image


# Calculate average CLIP Score
def calculate_average_clip_score(image_folder, text_prompt):
    scores = []
    for image_name in os.listdir(image_folder):
        if image_name.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, image_name)
            score = calculate_clip_score(image_path, text_prompt)
            scores.append(score)
            print(f"CLIP Score for {image_name}: {score}")
    average_score = sum(scores) / len(scores) if scores else 0
    print(f"Average CLIP Score: {average_score}")
    return average_score


def extract_frame():
    video_path = "sample_results/sampled_video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    frame_filename = f"sample_results/frame"
    if not os.path.exists(frame_filename):
        os.makedirs(frame_filename)
    frame_count = 0
    while True:
        # 读取下一帧
        ret, frame = cap.read()

        # 如果帧读取失败（视频结束），退出循环
        if not ret:
            break

        # 保存帧为图像文件
        frame_filename = f"sample_results/frame/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"Total frames extracted: {frame_count}")


if __name__ == "__main__":
    extract_frame()
    # 加载 CLIP 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义路径和文本描述
    real_images_path = "sample_results/image_prompt.png"  # 真实图像路径
    generated_images_path = "sample_results/frame"  # 生成图像路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_prompt = "panda on the sky"  # 文本指导
    calculate_average_clip_score(generated_images_path, text_prompt)
    # real_images = "sample_results/real_image"
    # img_path = "sample_results/generated_image"
    # calculate_fid(real_images, img_path, device)