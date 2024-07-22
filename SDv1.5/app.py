from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
import os

app = Flask(__name__)

model_id = "runwayml/stable-diffusion-v1-5"
model_path = "./diffusers"

# 모델이 이미 존재하는지 확인
if not os.path.exists(model_path):
    print("모델을 다운로드합니다. 이 작업은 시간이 걸릴 수 있습니다...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.save_pretrained(model_path)
    print("모델 다운로드 완료!")

# 기존의 모델 로드 코드
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe = pipe.to("cpu")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data["prompt"]

    # 이미지 생성
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=50).images[0]

    # 이미지 저장
    save_path = "static/generated_images"
    os.makedirs(save_path, exist_ok=True)
    image_filename = f"generated_{hash(prompt)}.png"
    image_path = os.path.join(save_path, image_filename)
    image.save(image_path)

    # URL 반환
    image_url = f"/static/generated_images/{image_filename}"
    return jsonify({"image_url": image_url})


if __name__ == "__main__":
    app.run(debug=True)
