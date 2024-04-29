from flask import Flask, request, jsonify, render_template
from shixian import predict
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 保存上传的文件到指定路径
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # 调用图片分析函数
    result = predict(file_path)

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
