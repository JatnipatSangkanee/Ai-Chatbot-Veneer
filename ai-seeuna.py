import pandas as pd
import faiss
import numpy as np
from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer, util
import requests
import json

# ตั้งค่าโมเดล SentenceTransformer
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# เชื่อมต่อ Neo4j
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "123456789_GG")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()

# ดึงข้อมูลข้อความจากฐานข้อมูล Neo4j สำหรับ Greetings
cypher_query_greeting = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
greeting_replies = {}
results = run_query(cypher_query_greeting)
for record in results:
    greeting_corpus.append(record['name'])
    greeting_replies[record['name']] = record['reply']

greeting_corpus = list(set(greeting_corpus))  # เอาข้อความมาใส่ใน corpus
print(greeting_corpus)

# ดึงข้อมูลข้อความจากฐานข้อมูล Neo4j สำหรับ Questions
cypher_query_question = '''
MATCH (n:Question) RETURN n.question as question, n.answer as answer;
'''
question_corpus = []
question_replies = {}
results = run_query(cypher_query_question)
for record in results:
    question_corpus.append(record['question'])
    question_replies[record['question']] = record['answer']

question_corpus = list(set(question_corpus))  # เอาข้อความมาใส่ใน corpus
print(question_corpus)
# Ollama API endpoint (assuming you're running Ollama locally)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

def llama_generate_response(prompt):
    # Prepare the request payload for the supachai/llama-3-typhoon-v1.5 model
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  # Adjust model name as needed
        "prompt": prompt+"ตอบไม่เกิน 20 คำและเป็นภาษาไทยเท่านั้น",
        "stream": False,
        "max_tokens": 60  # Limit the response to 100 tokens
    }

    # Send the POST request to the Ollama API
    response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        response_data = response.text
        data = json.loads(response_data)
        decoded_text = data.get("response", "No response found.")
        return decoded_text
    else:
        # Handle errors
        print(f"Failed to get a response: {response.status_code}, {response.text}")
        return "Error occurred while generating response."


# ฟังก์ชันคำนวณและหาข้อความตอบกลับ
def compute_response(sentence):
    # รวมเวกเตอร์จาก greeting และ question corpus
    combined_corpus = greeting_corpus + question_corpus
    combined_vec = model.encode(combined_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    
    # คำนวณ cosine similarities
    scores = util.cos_sim(combined_vec, ask_vec)
    scores_list = scores.tolist()
    scores_np = np.array(scores_list)
    
    max_score_index = np.argmax(scores_np)
    max_score = scores_np[max_score_index]
    
    # ตรวจสอบความเหมือนกับข้อความใน Neo4j
    if max_score > 0.6:
        # ถ้าความเหมือนมากกว่า 0.5 ให้ใช้ข้อความจาก Neo4j
        if max_score_index < len(greeting_corpus):
            match_text = greeting_corpus[max_score_index]
            response_msg = greeting_replies[match_text]
        else:
            match_text = question_corpus[max_score_index - len(greeting_corpus)]
            response_msg = question_replies[match_text]
    else:
        # ถ้าความเหมือนต่ำกว่า 0.5 ให้ใช้ Ollama API
        prompt = f"คำถาม: {sentence}\nคำตอบ:"
        response_msg = llama_generate_response(prompt)
        response_msg += "\n(ข้อความนี้ตอบโดย Ollama)"  # เพิ่มข้อความแจ้งว่าใช้ Ollama


    return response_msg


# สร้าง Flask app
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)  # รับข้อมูลจาก Line API
    try:
        json_data = json.loads(body)  # แปลงข้อมูลที่รับมาเป็น JSON
        access_token = '/1KhrjmLLx0kGUAfFHllqCzDSKCvvddJ00CRbrYn4KRb+aeK/yLnL+Viu75LHXbBHfhq+yTV20XGyPNBle9Axy5VFJxMgGo2TosPgf10V9TiAaVWkuG+teX5NsZlmBMMpUioSOHZ+vUFeE0R9YfZngdB04t89/1O/w1cDnyilFU='
        secret = '734f5114c2311c942789b2a6569d98b1'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        # ข้อความที่ได้รับจากผู้ใช้
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        
        # คำนวณและค้นหาข้อความตอบกลับ
        response_msg = compute_response(msg)
        
        # ส่งข้อความตอบกลับไปยัง Line
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        print(msg, tk)
    except Exception as e:
        print(f"Error: {e}")
        print(body)  # ในกรณีที่เกิดข้อผิดพลาด

    return 'OK'

if __name__ == '__main__':
    # For Debugging
    app.run(port=5000)
