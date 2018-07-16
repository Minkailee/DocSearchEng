from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from werkzeug.utils import secure_filename
from scipy import spatial
import os
import flask
from bs4 import BeautifulSoup
import re
import Feature_Vector as fv
import numpy
from sklearn.externals import joblib
import lsh


filelist = []
countlist = []
rootdir = "./static/9900/filenames.txt"
with open(rootdir,'r') as f:
    while 1:
        lines = f.readline()
        if not lines:
            break
        filelist.append(lines.strip("\n"))

words_set = fv.text_read("static/9900/words_set")
# base_vector = numpy.load(open())
estimator = joblib.load("static/9900/train_model.m")
app = Flask(__name__)
results = []

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/s')
def search():
    global results
    if request.method == 'GET':
        if 'wd' in request.args.keys():
            wd = request.args.get('wd')
            input_vector = fv.get_CompressedWordInputVector(wd, words_set, estimator)
            if input_vector == []:
                results = []
            else:
                results = get_results(input_vector)
            return flask.render_template('Result.html', result_list = results, page_index = 0)
        elif 'page_index' in request.args.keys():
            new_index = int(request.args.get('page_index'))
            return flask.render_template('Result.html', result_list = results, page_index = new_index)


def get_results(input_vector):
    global filelist
    dataset = numpy.load("static/9900/final_data.npy")

    keyno = lsh.search(dataset, input_vector[0], 50)
    similarity = []
    for k in keyno:
        similarity.append((1 - spatial.distance.cosine(input_vector[0], dataset[k])))
    # print(similarity)
    # print(keyno)
    result_list = [filelist[int(i)] for i in keyno]
    results = []
    count = 0
    for filename in result_list:
        # print(filename)
        soup = BeautifulSoup(open("static/NSWSC/" + filename, encoding="ISO-8859-1"), "lxml")
        s = str(soup.h2)
        # print(s)
        s = re.sub(r'<[^<>]*>', ' ', s).replace("\n", "")
        date = ""
        s1 = s[::-1]
        flag = False
        length = 0
        for x in s1:
            length += 1
            if x == ")":
                flag = True
                continue
            elif x == "(":
                flag = False
                break
            if flag:
                date += x
        date = date[::-1].split()
        # print(date)
        if len(date[0]) == 1:
            date[0] = "0" + date[0]
        if date[1].lower() == "january":
            date[1] = "01"
        elif date[1].lower() == "february":
            date[1] = "02"
        elif date[1].lower() == "march":
            date[1] = "03"
        elif date[1].lower() == "april":
            date[1] = "04"
        elif date[1].lower() == "may":
            date[1] = "05"
        elif date[1].lower() == "june":
            date[1] = "06"
        elif date[1].lower() == "july":
            date[1] = "07"
        elif date[1].lower() == "august":
            date[1] = "08"
        elif date[1].lower() == "september":
            date[1] = "09"
        elif date[1].lower() == "october":
            date[1] = "10"
        elif date[1].lower() == "november":
            date[1] = "11"
        elif date[1].lower() == "december":
            date[1] = "12"
        date = date[0] + "/" + date[1] + "/" + date[2]
        results.append([filename, s[:-length].strip(), date, int(similarity[count]*100)])
        results = sorted (results, key=lambda x: x[3], reverse=True)
        count += 1
    return results


@app.route('/f',methods=['GET', 'POST'])
def upload():
    global results
    if flask.request.method == 'GET':
        return flask.render_template('uploading.html')
    else:
        file=flask.request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/txt', secure_filename(file.filename))
        file.save(upload_path)
        input_vector = fv.get_CompressedFileInputVector(upload_path, words_set, estimator)
        if input_vector == []:
            results = []
        else:
            results = get_results(input_vector)
        return flask.render_template('Result.html', result_list = results, page_index = 0)

@app.route("/download/<path:filename>" )
def downloader(filename):
    dirpath = os.path.join(app.root_path, 'static/NSWSC')
    return send_from_directory(dirpath, filename, as_attachment=True)

@app.route("/showfile/<path:filename>")
def displayer(filename):
    return flask.render_template('showfile.html', Content = filename)

@app.route('/rs')
def result():
    if flask.request.method=='GET':
        return flask.render_template('Result.html')

if __name__ == '__main__':
    app.run()
