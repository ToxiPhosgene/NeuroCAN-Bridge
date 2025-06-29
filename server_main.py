from flask import Flask, request

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def handle_data():
    data = request.json
    print("Получены данные:", data)
    return {"status": "OK"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)