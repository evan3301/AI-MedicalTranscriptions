import sys, torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(model_dir, text):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    prob = out.logits.softmax(-1)[0]
    pred_id = int(prob.argmax())
    id2label = mdl.config.id2label
    label = id2label[str(pred_id)] if isinstance(list(id2label.keys())[0], str) else id2label[pred_id]
    return {"label": label, "probs": { (id2label[str(i)] if isinstance(list(id2label.keys())[0], str) else id2label[i]): float(prob[i]) for i in range(prob.size(0))}}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/classify/predict.py <model_dir> <text>")
        sys.exit(1)
    model_dir, text = sys.argv[1], " ".join(sys.argv[2:])
    print(json.dumps(predict(model_dir, text), indent=2))
