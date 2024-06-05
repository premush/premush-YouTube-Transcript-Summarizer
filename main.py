from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from http import HTTPStatus
import json
app = Flask(__name__)
CORS(app)
extension_open = False

def transcript_to_text(transcript):
    formatter = JSONFormatter()
    json_formatted = formatter.format_transcript(transcript)
    parsed_transcript = json.loads(json_formatted)
    speech_text = " ".join([item['text'] for item in parsed_transcript])
    return speech_text


def text_summary_t5_tokenizer(script):
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base",model_max_length=1024)
    inputs = tokenizer.encode("summarize: " + script, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=400, min_length=100,  length_penalty=2.0, num_beams=4,
                             early_stopping=True)
    return tokenizer.decode(outputs[0])


@app.errorhandler(404)
@app.errorhandler(500)
def handle_error(e):
    error_code = e.code
    error_message = HTTPStatus(e.code).phrase
    return f'Bad Request! Error: {error_code}, {error_message}'


@app.route('/api/summarize/t5-tokenizer', methods=['GET'])
def summarize_t5_tokenizer():
    youtube_url = request.args.get('youtube_url')
    if not youtube_url:
        return jsonify({"error": "Missing 'youtube_url' parameter"}), 400
    video_id = youtube_url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    script = transcript_to_text(transcript)
    summary = text_summary_t5_tokenizer(script)
    return jsonify({"summary": summary})


if __name__ == '__main__':
    app.run(debug=True)
