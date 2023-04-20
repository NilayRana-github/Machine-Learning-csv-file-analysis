# Project backup branch

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
app.secret_key = 'mysecretkey'
app.config['DATASET_FOLDER'] = 'dataset'


@app.route('/')
def home():
    return render_template("home_page.html")


def process_data(file_path):
    data = pd.read_csv(file_path)

    head = data.head().to_html(classes='table table-striped')
    tail = data.tail().to_html(classes='table table-striped')
    info = data.info
    size = data.shape

    missing_value_before = data.isna().sum()

    if data.isnull().values.any():
        data.dropna(inplace=True)
    missing_value_after = data.isna().sum()

    duplicate_value_before = data.duplicated().sum()

    if data.duplicated().any():
        data.drop_duplicates(ignore_index=False, inplace=True)
    duplicate_value_after = data.duplicated().sum()

    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']):
        data[col] = le.fit_transform(data[col])

    correlation = data.corr().to_html(classes='table table-striped')
    describe = data.describe().to_html(classes='table table-striped')
    columns = list(data.columns)

    return data, head, tail, info, size, missing_value_before, missing_value_after, duplicate_value_before, duplicate_value_after, describe, correlation, columns


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            csv_file = request.files['csv_file']
            file_path = os.path.join(app.config['DATASET_FOLDER'], csv_file.filename)
            csv_file.save(file_path)
            session['file_path'] = file_path

            data, head, tail, info, size, missing_value_before, missing_value_after, duplicate_value_before, duplicate_value_after, describe, correlation, columns = process_data(
                file_path)

            return render_template('index.html', uploaded=True, head=head, tail=tail, info=info, size=size,
                                   missing_value_before=missing_value_before,
                                   missing_value_after=missing_value_after,
                                   duplicate_value_before=duplicate_value_before,
                                   duplicate_value_after=duplicate_value_after,
                                   describe=describe,
                                   correlation=correlation,
                                   columns=columns)
        else:
            column_name = request.form.get('column_name')
            algorithm = request.form.get('algorithm')
            file_path = session.get('file_path')

            if file_path is None or not os.path.isfile(file_path):
                return render_template('index.html')

            data, head, tail, info, size, missing_value_before, missing_value_after, duplicate_value_before, duplicate_value_after, describe, correlation, columns = process_data(
                file_path)

            print("Printing info in else after label encoding")
            data.info()

            print("Column name")
            print(column_name)

            x = data.drop(column_name, axis=1)
            y = data[column_name]

            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

            if algorithm == 'Decision Tree':
                model = DecisionTreeClassifier()
            elif algorithm == 'Logistic Regression':
                model = LogisticRegression()
            elif algorithm == 'SVC':
                model = SVC()
            elif algorithm == 'Random Forest':
                model = RandomForestClassifier()
            else:
                raise ValueError('Invalid algorithm specified')

            print(model)

            model.fit(xtrain, ytrain)
            y_pred = model.predict(xtest)

            confusion_mat = confusion_matrix(ytest, y_pred)
            report = classification_report(ytest, y_pred)
            accuracy = accuracy_score(ytest, y_pred) * 100

            # Convert confusion matrix to Pandas DataFrame
            confusion_mat_df = pd.DataFrame(confusion_mat, columns=np.unique(ytest), index=np.unique(ytest))

            os.remove(file_path)
            session.pop('csv_file_name', None)  # remove file name from session

            return render_template('index.html', uploaded=True, head=head, tail=tail, info=info, size=size,
                                   missing_value_before=missing_value_before,
                                   missing_value_after=missing_value_after,
                                   duplicate_value_before=duplicate_value_before,
                                   duplicate_value_after=duplicate_value_after,
                                   describe=describe,
                                   correlation=correlation,
                                   columns=columns,
                                   confusion_mat=confusion_mat_df.to_html(), report=report, accuracy=accuracy)

        return render_template('index.html', uploaded=False)
    else:
        return render_template('index.html', uploaded=False)


@app.route('/regr', methods=['GET', 'POST'])
def regr():
    if request.method == 'POST':
        if request.files:
            csv_file = request.files['csv_file']
            file_path = os.path.join(app.config['DATASET_FOLDER'], csv_file.filename)
            csv_file.save(file_path)
            session['file_path'] = file_path

            data, head, tail, info, size, missing_value_before, missing_value_after, duplicate_value_before, duplicate_value_after, describe, correlation, columns = process_data(
                file_path)

            return render_template('regr.html', uploaded=True, head=head, tail=tail, info=info, size=size,
                                   missing_value_before=missing_value_before,
                                   missing_value_after=missing_value_after,
                                   duplicate_value_before=duplicate_value_before,
                                   duplicate_value_after=duplicate_value_after,
                                   describe=describe,
                                   correlation=correlation,
                                   columns=columns)
        else:
            column_name = request.form.get('column_name')
            algorithm = request.form.get('algorithm')
            file_path = session.get('file_path')

            if file_path is None or not os.path.isfile(file_path):
                return render_template('regr.html')

            data, head, tail, info, size, missing_value_before, missing_value_after, duplicate_value_before, duplicate_value_after, describe, correlation, columns = process_data(
                file_path)

            print("Printing info in else after label encoding")
            data.info()

            print("Column name")
            print(column_name)

            x = data.drop(column_name, axis=1)
            y = data[column_name]

            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

            if algorithm == 'Decision Tree Regression':
                model = DecisionTreeRegressor()
            elif algorithm == 'Random Forest Regression':
                model = RandomForestRegressor()
            elif algorithm == 'SVC Regression':
                model = SVR()
            elif algorithm == 'Linear Regression':
                model = LinearRegression()
            else:
                raise ValueError('Invalid algorithm specified')

            print(model)

            model.fit(xtrain, ytrain)
            y_pred = model.predict(xtest)

            R2_score = r2_score(ytest, y_pred) * 100
            Mse = mean_squared_error(ytest, y_pred)
            Rmse = np.sqrt(mean_squared_error(ytest, y_pred))
            Mae = mean_absolute_error(ytest, y_pred)

            os.remove(file_path)
            session.pop('csv_file_name', None)  # remove file name from session

            return render_template('regr.html', uploaded=True, head=head, tail=tail, info=info, size=size,
                                   missing_value_before=missing_value_before,
                                   missing_value_after=missing_value_after,
                                   duplicate_value_before=duplicate_value_before,
                                   duplicate_value_after=duplicate_value_after,
                                   describe=describe,
                                   correlation=correlation,
                                   columns=columns,
                                   R2_score=R2_score, Mse=Mse, Rmse=Rmse, Mae=Mae)
    else:
        return render_template('regr.html', uploaded=False)


if __name__ == '__main__':
    app.run(debug=True)
