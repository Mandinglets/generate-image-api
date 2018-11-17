from flask import Flask, request, jsonify
from flask_cors import CORS
from serve import get_model_api
import numpy as np

from flask import send_file
import scipy.misc
import hashlib

import os

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default

@app.route('/')
def index():
    return "Index API"

@app.errorhandler(404)
def url_error(e):
    return f"""
    Wrong URL!
    <pre>{e}</pre>""", 404

@app.errorhandler(500)
def server_error(e):
    return f"""
    An internal error occurred: <pre>{e}</pre>
    See logs for full stacktrace.
    """, 500

@app.route('/api/test', methods=['POST'])
def api():
    r = request
    data = r.data
    z = np.fromstring(data, dtype=np.float64).reshape((1, 100))

    output_data = get_model_api(z)
    data_path = hashlib.sha256(output_data.tostring()).hexdigest() + '.png'
    scipy.misc.imsave(f'img/{data_path}', output_data[0, :, :, :])
    return send_file(os.path.join('img', data_path), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
