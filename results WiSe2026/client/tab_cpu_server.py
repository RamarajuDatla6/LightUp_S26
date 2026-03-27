

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# 🔥 THIS LINE IS CRITICAL
CORS(app, resources={r"/*": {"origins": "*"}})

latest_cpu = None


@app.route("/tab_cpu", methods=["POST"])
def receive_cpu():
    global latest_cpu
    data = request.get_json()
    latest_cpu = data.get("cpu")
    return jsonify({"status": "ok"})


@app.route("/get_cpu", methods=["GET"])
def get_cpu():
    return jsonify({"cpu": latest_cpu})


if __name__ == "__main__":
    print("Tab CPU server running on http://localhost:5050")
    app.run(host="127.0.0.1", port=5050)

