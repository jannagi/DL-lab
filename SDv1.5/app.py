from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

app = Flask(__name__)

# Stable Diffusion 모델 설정
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_model_path = "./diffusers"

if not os.path.exists(sd_model_path):
    print(
        "Stable Diffusion 모델을 다운로드합니다. 이 작업은 시간이 걸릴 수 있습니다..."
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id, torch_dtype=torch.float16
    )
    pipe.save_pretrained(sd_model_path)
    print("Stable Diffusion 모델 다운로드 완료!")

# LLaMA 모델 설정
llama_model_id = "meta-llama/Meta-Llama-3-8B"
llama_model_path = "./llama3"

if not os.path.exists(llama_model_path):
    print("LLaMA 모델을 다운로드합니다. 이 작업은 시간이 걸릴 수 있습니다...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(llama_model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        token=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer.save_pretrained(llama_model_path)
    model.save_pretrained(llama_model_path)
    print("LLaMA 모델 다운로드 완료!")

tokenizer = None
model = None
pipe = None


def load_llama_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("LLaMA 모델을 로드합니다...")
        tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            llama_model_path, device_map="balanced"
        )


def load_sd_model():
    global pipe
    if pipe is None:
        print("Stable Diffusion 모델을 로드합니다...")
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_path, torch_dtype=torch.float16, device_map="balanced"
        )
        pipe.enable_attention_slicing()


def unload_llama_model():
    global tokenizer, model
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    torch.cuda.empty_cache()


def unload_sd_model():
    global pipe
    if pipe is not None:
        del pipe
        pipe = None
    torch.cuda.empty_cache()


def generate_prompt_variations(prompt, num_variations=3):
    load_llama_model()

    system_message = "You are a creative AI assistant specialized in generating visually descriptive variations of image prompts. Your task is to create unique and interesting variations based on the given prompt, focusing on visual details that can be clearly represented in an image."

    user_message = f"""Given the following image prompt, generate {num_variations} unique and visually descriptive variations. Each variation should be a concise, specific description that could be used directly as an input for an image generation model. Focus on visual elements, colors, composition, and style. Keep each variation under 15 words.

Original prompt: {prompt}

Please provide your variations in the following format:
1. [First variation]
2. [Second variation]
3. [Third variation]

Example:
Original prompt: A serene landscape
1. Misty mountains at dawn, purple sky, reflection in a calm lake
2. Golden wheat field under a dramatic stormy sky, single oak tree
3. Tranquil Japanese garden with a red bridge, cherry blossoms, koi pond

Now, provide your {num_variations} variations for the given prompt:"""

    # llama3 프롬프트 변형 생성을 위한 세부 지시사항
    input_text = f"{system_message}\n\nHuman: {user_message}\n\nAssistant:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # 프롬프트 변형 조정
    output = model.generate(
        input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 출력에서 변형 프롬프트 추출
    all_variations = [
        line.split(".", 1)[1].strip()
        for line in response.split("\n")
        if line.strip().startswith(("1.", "2.", "3."))
    ]

    variations = all_variations[-num_variations:]

    # 변형 수가 부족한 경우 기존 프롬프트로 대체
    while len(variations) < num_variations:
        variations.append(prompt)

    print("\nResponse:", response, end="\n\n")
    print("\nGenerated variations:", variations, end="\n\n")
    unload_llama_model()
    return variations


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.json
    original_prompt = data["prompt"]

    # 프롬프트 변형 생성
    variations = generate_prompt_variations(original_prompt)

    load_sd_model()
    images = []
    for i, prompt in enumerate([original_prompt] + variations):
        # 이미지 생성
        image = pipe(prompt).images[0]

        # 이미지 저장
        save_path = "static/generated_images"
        os.makedirs(save_path, exist_ok=True)
        image_filename = f"generated_{hash(prompt)}_{i}.png"
        image_path = os.path.join(save_path, image_filename)
        image.save(image_path)

        # URL 저장
        image_url = f"/static/generated_images/{image_filename}"
        images.append({"prompt": prompt, "image_url": image_url})

    unload_sd_model()
    return jsonify({"images": images})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
