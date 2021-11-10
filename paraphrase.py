from flask import Flask, request, jsonify


from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt")
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

app = Flask(__name__)

@app.route("/")
def home():

    return "<h1>Sanity Check.  The API is up and running.</h1>"

@app.route("/api/paraphrase/", methods=['POST'])
def paraphrase():

    input = request.get_json()['input']
    rewritten = get_response(input, 1, 10)
    return jsonify(isError=False, message="Success", data=rewritten), 200

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000)
