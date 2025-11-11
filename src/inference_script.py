import argparse
import torch
import mlflow
import mlflow.pytorch
import os
from PIL import Image
from src.utils import infer_large_image

def main():
    parser = argparse.ArgumentParser(description="running inference for deblurring")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_folder", type=str,required=True)
    parser.add_argument("--date", type=str, required=True)

    args = parser.parse_args()

    model = mlflow.pytorch.load_model(args.model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(args.output_folder, exist_ok=True)

    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                input_path = os.path.join(args.input_folder, filename)
                output_path = os.path.join(args.output_folder, f"{args.model_name}_{args.date}_{filename}")

                file_image = Image.open(input_path).convert("RGB")
                
                # get the output
                generated_image = infer_large_image(model=model, img_pil=file_image, device=device, tile_size=256, overlap=64)

                generated_image.save(output_path)
            except Exception as e:
                print(f"Failed with exception: {e}")
                continue


if __name__ == "__main__":
    main()
