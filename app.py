from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)


@app.route('/guide', methods=['POST'])
def composition_guide():
    image = request.json['image']
    # f = request.files['file']
    print(image)
    print(request.json)
    return 'SUCCESS'

@app.route('/test')
def hello():
    return 'python'

if __name__ == '__main__':
    app.run(port=8000)
