import os
from app import app, models, sr_net
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, render_template, send_from_directory


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        mode = request.form['mode']
        print(request.form)
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('app', app.config['UPLOAD_FOLDER'], filename))

    
        return redirect(url_for('result',
                                filename=filename, mode=mode))
    
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename, mode=None):
    mode = request.args.get('mode')
    print(mode)
    if mode == "bench":
        print('bench')
        result_path = sr_net.get_bench_image('./app/upload/', filename, 4)
        if result_path[-1] == 'fail':
            result_path = sr_net.get_bench_image('./app/upload/','fail.png',4)
    elif mode == '4':
        print(4)
        result_path = sr_net.get_sr_image('./app/upload/', filename, 4)
        if result_path[-1] == 'fail':
            result_path = sr_net.get_sr_image('./app/upload/','fail.png',4)
    elif mode == '2j':
        print(4)
        result_path = sr_net.get_sr_image_jpeg('./app/upload/', filename, 4)
        if result_path[-1] == 'fail':
            result_path = sr_net.get_sr_image_jpeg('./app/upload/','fail.png',4)
    elif mode == '2':
        print(2)
        result_path = sr_net.get_sr_image('./app/upload/', filename, 2)
        if result_path[-1] == 'fail':
            result_path = sr_net.get_sr_image('./app/upload/','fail.png',2)
    else :
        result_path = ['lr_4x_'+filename, 'bicubic_4x_'+filename, 'sr_4x_'+filename]
    return render_template('result.html',filename = [filename] + result_path)

@app.route('/upload/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.errorhandler(413)
def error413(e):
    return "File is too big", 413