from flask import Flask, request, render_template
from transformers import BertTokenizer, BertConfig, BertForQuestionAnswering
import torch

app = Flask(__name__)

# 1. 토크나이저와 설정 로드
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
config = BertConfig.from_pretrained("beomi/kcbert-base")

# 2. 모델 생성 및 체크포인트 불러오기
model = BertForQuestionAnswering(config)
ckpt = torch.load("checkpoint/model.ckpt", map_location=torch.device("cpu"))
model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}, strict=False)
model.eval()

# 3. 웹 인터페이스
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    question = ""
    context = ""

    if request.method == "POST":
        question = request.form["question"]
        context = request.form["context"]

        inputs = tokenizer.encode_plus(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1

        if start <= end:
            answer = tokenizer.decode(
                inputs["input_ids"][0][start:end],
                skip_special_tokens=True
            )
        else:
            answer = "❗ 적절한 답변을 찾을 수 없습니다."

    return render_template("index.html", answer=answer, question=question, context=context)

if __name__ == "__main__":
    app.run(debug=True)
