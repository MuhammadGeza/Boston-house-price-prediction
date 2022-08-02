import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import base64
from scipy.stats import median_abs_deviation, skew, kurtosis, pearsonr
import pickle

@st.cache
def decode_img(dir_img):
    with open(dir_img, "rb") as image_file:
        byt = base64.b64encode(image_file.read())
    return byt.decode()

@st.cache
def load_df(dir):
    return pd.read_csv(dir)

@st.cache
def sample_data(df, n):
    return df.sample(n)

def atribut_information():
    st.markdown("""
            <div class="container border border-dark">
                <div class="row mt-2 Viga" align="justify">
                    <p class="mb-auto">Input features in order:</p>
                    <ul>
                        <li><span style="color:#00a86b;">CRIM</span> : per capita crime rate by town</li>
                        <li><span style="color:#00a86b;">ZN</span> : proportion of residential land zoned for lots over 25,000 sq.ft.</li>
                        <li><span style="color:#00a86b;">INDUS</span> : proportion of non-retail business acres per town</li>
                        <li><span style="color:#00a86b;">CHAS</span> : Charles River dummy variable (1 if tract bounds river; 0 otherwise)</li>
                        <li><span style="color:#00a86b;">NOX</span> : nitric oxides concentration (parts per 10 million) [parts/10M]</li>
                        <li><span style="color:#00a86b;">RM</span> : average number of rooms per dwelling</li>
                        <li><span style="color:#00a86b;">AGE</span> : proportion of owner-occupied units built prior to 1940</li>
                        <li><span style="color:#00a86b;">DIS</span> : weighted distances to five Boston employment centres</li>
                        <li><span style="color:#00a86b;">RAD</span> : index of accessibility to radial highways</li>
                        <li><span style="color:#00a86b;">TAX</span> : full-value property-tax rate per $10,000 [$/10k]</li>
                        <li><span style="color:#00a86b;">PTRATIO</span> : pupil-teacher ratio by town</li>
                        <li><span style="color:#00a86b;">B</span> : The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town</li>
                        <li><span style="color:#00a86b;">LSTAT</span> : % lower status of the population</li>
                    </ul>
                    <p class="mb-auto">Output variable:</p>
                    <ul>
                        <li><span style="color:#00a86b;">MEDV</span> : Median value of owner-occupied homes in $1000's [k$]</li>
                    </ul>
                </div>
            </div>""", unsafe_allow_html=True)

def dataset_statistics(df):
    st.markdown(f"""
            <div class="container border border-dark">
                <div class="row">
                    <div class="col-md-6">
                        <table class="table table-hover mt-3">
                                <tr>
                                    <th scope="row">Number of variabels</th>
                                    <td>{len(df.columns)}</td>
                                </tr>
                                <tr>
                                    <th scope="row">Number of observations</th>
                                    <td>{len(df)}</td>
                                </tr>
                                <tr>
                                    <th scope="row">Missing cells</th>
                                    <td>{sum(df.isna().sum())}</td>
                                </tr>
                                <tr>
                                    <th scope="row">Missing cells (%)</th>
                                    <td>{sum(df.isna().mean()) * 100} %</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <table class="table table-hover mt-3">
                            <tbody>
                                <tr>
                                    <th scope="row">Duplicate rows</th>
                                    <td>{sum(df.duplicated())}</td>
                                </tr>
                                <tr>
                                    <th scope="row">Duplicate rows (%)</th>
                                    <td>{sum(df.duplicated()) / len(df)} %</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_png = buffer.getvalue()
    graph = base64.b64encode(img_png).decode('utf-8')
    buffer.close()
    return graph

def distplot(df):
    plt.figure(figsize=(2, 5))
    sns.displot(df, height=2.5, kde=True)  
    graph = get_graph()
    return graph

def count_plot(df):
    plt.figure(figsize=(6, 7))
    sns.countplot(df, saturation=1) 
    plt.xticks(rotation=45, ha='right')
    graph = get_graph()
    return graph

def tabel(df):
    html = """"""
    d = df.value_counts().sort_index().to_dict()
    for i, j in d.items():
        html += f"""<tr><td>{i}</td><td>{j}</td><td>{round(j / len(df) * 100, 3)} %</td></tr>"""
    return html

def variabels_overview(df):
    categoric = ['CHAS', 'RAD']
    for i in df.columns:
        if i not in categoric:
            image = distplot(df[i])
            st.markdown(f"""
                <div class="container border border-dark">
                    <div class="row">
                            <div class="col-md-12 text-center">
                                <h3 clas="" style="color: #337AB7; font-family: Viga;">{i} <br> <p style="color: black;">Numeric</p></h3>
                            </div>
                        </div>
                    <div class="row border">
                        <div class="col-lg-4 col-md-6">
                            <table class="table mt-3">
                            <tbody>
                            <tr>
                                <th scope="row">Unique</th>
                                <td>{df[i].nunique()}</td>
                            </tr>
                            <tr>
                                <th scope="row">Unique (%)</th>
                                <td>{round(df[i].nunique() / len(df) * 100, 3)} %</td>
                            </tr>
                            <tr>
                                <th scope="row">Missing</th>
                                <td>{sum(df[i].isna())}</td>
                            </tr>
                            <tr>
                                <th scope="row">Missing (%)</th>
                                <td>{round(sum(df[i].isna()) / len(df), 3)} %</td>
                            </tr>
                            <tr>
                                <th scope="row">Zeros</th>
                                <td>{sum(df[i]==0)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Zeros (%)</th>
                                <td>{round(sum(df[i]==0) / len(df) * 100, 3)} %</td>
                        </tbody>
                        </table>
                        </div>
                        <div class="col-lg-4 col-md-6">
                        <table class="table table-hover mt-3">
                            <tbody>
                            <tr>
                                <th scope="row">Mean</th>
                                <td>{round(df[i].mean(), 3)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Median</th>
                                <td>{round(df[i].median(), 3)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Minimum</th>
                                <td>{df[i].min()}</td>
                            </tr>
                            <tr>
                                <th scope="row">Maximum</th>
                                <td>{df[i].max()}</td>
                            </tr>
                        </tbody>
                        </table>
                        </div>
                        <div class="col-lg-4 col-md-12">
                            <div class="justify-content-center text-center">
                                <img src="data:image/png;base64,{image}">
                            </div>
                        </div>
                        </div>
                        <div class="row mt-2">
                        <div class="col-md-6 mb-2">
                            <p class="h4 text-center">Quantile statistics</p>
                            <table class="table table-hover">
                            <tbody>
                            <tr>
                                <th scope="row">Minimum</th>
                                <td>{df[i].min()}</td>
                            </tr>
                            <tr>
                                <th scope="row">5-th percentile</th>
                                <td>{round(df[i].quantile(.05),4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Q1</th>
                                <td>{round(df[i].quantile(.25),4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Median</th>
                                <td>{round(df[i].quantile(.5), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Q3</th>
                                <td>{round(df[i].quantile(.75),4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">95-th percentile</th>
                                <td>{round(df[i].quantile(.95),4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Interquartile range (IQR)</th>
                                <td>{round(df[i].quantile(.75) - df[i].quantile(.25), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Maximum</th>
                                <td>{df[i].max()}</td>
                            </tr>
                            <tr>
                                <th scope="row">Range</th>
                                <td>{round(df[i].max() - df[i].min(), 4)}</td>
                            </tr>
                        </tbody>
                        </table>
                        </div>
                        <div class="col-md-6 mb-2">
                        <p class="h4 text-center">Descriptive statistics</p>
                            <table class="table table-hover">
                            <tbody>
                            <tr>
                                <th scope="row">Variance</th>
                                <td>{round(df[i].var(), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Standard deviation</th>
                                <td>{round(df[i].std(), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Coefficient of variation (CV)</th>
                                <td>{round(df[i].std() / df[i].mean() * 100, 4)} %</td>
                            </tr>
                            <tr>
                                <th scope="row">Mean</th>
                                <td>{round(df[i].mean(), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Median Absolute Deviation (MAD)	</th>
                                <td>{round(median_abs_deviation(df[i]), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Mean Absolute Deviation (MAD)	</th>
                                <td>{round(df[i].mad(), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Skewness</th>
                                <td>{round(skew(df[i]), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Kurtosis</th>
                                <td>{round(kurtosis(df[i]), 4)}</td>
                            </tr>
                            <tr>
                                <th scope="row">Sum</th>
                                <td>{round(df[i].sum(), 4)}</td>
                            </tr>
                        </tbody>
                        </table>
                        </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            image = count_plot(df[i])
            st.markdown(f"""
                <div class="container border border-dark">
                    <div class="row">
                            <div class="col-md-12 text-center">
                                <h3 clas="" style="color: #337AB7; font-family: Viga;">{i} <br> <p style="color: black;">Categoric</p></h3>
                            </div>
                        </div>
                    <div class="row border">
                        <div class="col-lg-8">
                            <table class="table mt-3">
                            <tbody>
                            <tr>
                                <th scope="row">Count</th>
                                <td>{df[i].count()}</td>
                            </tr>
                            <tr>
                                <th scope="row">Unique</th>
                                <td>{df[i].nunique()}</td>
                            </tr>
                            <tr>
                                <th scope="row">Unique (%)</th>
                                <td>{round(df[i].nunique() / len(df) * 100, 3)} %</td>
                            </tr>
                            <tr>
                                <th scope="row">Missing</th>
                                <td>{sum(df[i].isna())}</td>
                            </tr>
                            <tr>
                                <th scope="row">Missing (%)</th>
                                <td>{round(sum(df[i].isna()) / len(df), 3)} %</td>
                            </tr>
                        </tbody>
                        </table>
                        </div>
                        <div class="col-lg-3">
                            <div class="justify-content-center text-center">
                                <img style="width: 270px; height: 250px;" src="data:image/png;base64,{image}">
                            </div>
                        </div>
                        </div>
                        <div class="row mt-2 justify-content-center">
                        <div class="col-md-8">
                            <table class="table mt-3">
                            <tbody>
                                <p class="h4 text-center">Common values</p>
                                <tr>
                                    <th scope="row">Value</th>
                                    <th scope="row">Count</th>
                                    <th scope="row">Frequency</th>
                                </tr>
                                <div>{tabel(df[i])}</div>
                            </tbody>
                            </table>
                        </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

def scatter(df, x, y):
    fig = plt.figure(figsize=(9, 4.5))
    plt.title(f'Correlation value (pearson): {pearsonr(df[x], df[y])[0]}')
    sns.scatterplot(x=x, y=y, data=df)
    graph = get_graph()
    return graph

def interactions(df):
    col1, col2 = st.columns(2)
    with col1:
        col1 = st.selectbox('X axis', df.columns)
    with col2:
        col2 = st.selectbox('Y axis', df.columns[::-1])

    image = scatter(df, col1, col2)
    st.markdown(f"""<div class="container border border-dark">
                    <div class="row text-center">
                        <div class="col-md-12 mb-3">
                            <img class="responsive" style="width: 100; height: 100;" src="data:image/png;base64,{image}">
                        </div>
                    </div>
                    </div>""", unsafe_allow_html=True)

attr = {
        'pearson': ['coolwarm', "<h3>Pearson's r</h3>The Pearson's correlation coefficient (<em>r</em>) is a measure of linear correlation between two variables. It's value lies between -1 and +1, -1 indicating total negative linear correlation, 0 indicating no linear correlation and 1 indicating total positive linear correlation. Furthermore, <em>r</em> is invariant under separate changes in location and scale of the two variables, implying that for a linear function the angle to the x-axis does not affect <em>r</em>.<br><br>To calculate <em>r</em> for two variables <em>X</em> and <em>Y</em>, one divides the covariance of <em>X</em> and <em>Y</em> by the product of their standard deviations."],
        'kendall': ['viridis', "<h3>Kendall's τ</h3>Similarly to Spearman's rank correlation coefficient, the Kendall rank correlation coefficient (<em>τ</em>) measures ordinal association between two variables. It's value lies between -1 and +1, -1 indicating total negative correlation, 0 indicating no correlation and 1 indicating total positive correlation. <br><br>To calculate <em>τ</em> for two variables <em>X</em> and <em>Y</em>, one determines the number of concordant and discordant pairs of observations. <em>τ</em> is given by the number of concordant pairs minus the discordant pairs divided by the total number of pairs."],
        'spearman': ['PRGn', "<h3>Spearman's ρ</h3>The Spearman's rank correlation coefficient (<em>ρ</em>) is a measure of monotonic correlation between two variables, and is therefore better in catching nonlinear monotonic correlations than Pearson's <em>r</em>. It's value lies between -1 and +1, -1 indicating total negative monotonic correlation, 0 indicating no monotonic correlation and 1 indicating total positive monotonic correlation.<br><br>To calculate <em>ρ</em> for two variables <em>X</em> and <em>Y</em>, one divides the covariance of the rank variables of <em>X</em> and <em>Y</em> by the product of their standard deviations."]
    }

def heatmap_plot(df, method):
    plt.figure(figsize=(7, 6))
    sns.heatmap(df.corr(method=method), annot=True, annot_kws={'size': 5.3}, cmap=attr[method][0], vmin=-1, vmax=1, linewidths=0.1, square=True)
    graph = get_graph()
    return graph
    
def correlations(df):
    method = st.selectbox('Choose method',['pearson', 'kendall', 'spearman'])
    image = heatmap_plot(df, method)
    st.markdown(f"""<div class="container border border-dark">
                    <div class="row">
                        <div class="col-md-8 mb-3 text-center">
                            <img class="responsive" src="data:image/png;base64,{image}">
                        </div>
                        <div class="col-md-4 mb-3" align="justify">
                            {attr[method][1]}
                        </div>
                    </div>
                    </div>""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model(dir):
    return pickle.load(open(dir, 'rb'))

def model_overview():
    st.markdown(f"""
            <div class="container border border-dark">
                <div class="row mt-2 Viga" align="justify">
                    <p>In this case, I use the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" class="decoration-none">RandomForestRegressor</a> model. But when training the model, I don't use all the data for training so I can evaluate the model with new data that has never been seen by previous model. I split the data using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" class="decoration-none">train_test_split</a> with <span style="color:#00a86b;">80%</span> training data and <span style="color:#00a86b;">20%</span> test data</p>
                </div>
            </div>""", unsafe_allow_html=True)

def model_cv_result(model):
    df = pd.DataFrame(model.cv_results_).loc[:, 'params':].sort_values('rank_test_score')[:10]
    st.write(df)

def model_parameters(model):
    html1 = """"""
    for i, j in model.param_grid.items():
        li = f"""<li>{i.split('__')[1]}: <span style="color:#00a86b;">{j}</span></li>"""
        html1 += li
    
    html2 = """"""
    for i, j in model.best_params_.items():
        li = f"""<li>{i.split('__')[1]}: <span style="color:#00a86b;">{j}</span></li>"""
        html2 += li
    st.markdown(f"""
            <div class="container border border-dark">
                <div class="row mt-2 Viga" align="justify">
                    <p class="mb-auto">Using a <a class="decoration-none" href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> with {model.cv} <a href="https://scikit-learn.org/stable/modules/cross_validation.html" class="decoration-none">cross validation</a>, the best combination of parameter values will be searched for the following parameters: </p>
                    <ul>{html1}</ul>
                    <p class="mb-auto">The best combination of parameters obtained with a score of <span style="color:#00a86b;">{round(model.best_score_, 6)*100} %</span>, with parameter:</p>
                    <ul>{html2}</ul>
                </div>
            </div>""", unsafe_allow_html=True)

def model_scores(model, X_train, X_test, y_train, y_test):
    st.markdown(f"""
            <div class="container border border-dark">
                <div class="row mt-2 Viga" align="justify">
                    <p class="mb-auto">This is the result of a score of {model.cv} <a href="https://scikit-learn.org/stable/modules/cross_validation.html" class="decoration-none">cross validation</a>, which is calculated using the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html" class="decoration-none">r2_score</a>:</p>
                    <ul>
                        <li>Best score of {model.cv} <a href="https://scikit-learn.org/stable/modules/cross_validation.html" class="decoration-none">cross validation</a>: <span style="color:#00a86b;">{round(model.best_score_, 6)*100} %</span></li>
                        <li>Score on training data: <span style="color:#00a86b;">{round(model.score(X_train, y_train), 6)*100} %</span></li>
                        <li>Score on test data: <span style="color:#00a86b;">{round(model.score(X_test, y_test), 6)*100} %</span></li>
                    </ul>
                </div>
            </div>""", unsafe_allow_html=True)

def model_pipeline(model):
    st.markdown(f"""
            <div class="container border border-dark">
                <div class="row mt-2 Viga" align="justify">
                    <p class="mb-1"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html" class="decoration-none">Pipelines</a> in machine learning are used to help automate learning workflows machine. They operate by allowing sequences of data to be transformed and correlated together in a model that can be tested and evaluated to achieve results, either positive or negative.</p>
                    <p class="mb-1"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html" class="decoration-none">Pipelines</a> can be used to connect multiple estimators into one. This is useful because there is often a fixed sequence of steps in data processing, such as feature selection, normalization, and classification.</p>
                    <p class="mb-auto"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html" class="decoration-none">Pipelines</a> are used for several purposes here:</p>
                    <ul>
                        <li>Convenience and encapsulation: You just need to call fit and predict once on your data to match the whole sequence estimator.</li>
                        <li>Shared parameter selection: You can search a grid over the parameters of all the estimators in the pipeline at once.</li>
                        <li>Security: Pipelines help avoid leaking statistics from your test data into models trained in cross-validation, ensuring that the same sample is used to train the transformer and predictor.</li>
                    </ul>
                    <p class="mb-auto">Many datasets contain features of different types, such as text, float, and date, where each type features require separate preprocessing or feature extraction steps. Often the easiest to preprocess the data before applying the scikit-learn method, for example using pandas. Processing your data before passing it to scikit-learn may be problematic for one reason following:</p>
                    <ul>
                        <li>Entering statistics from test data into the preprocessor renders cross-validation scores impossible reliable (known as data leakage), for example in the case of scaling or inserting values missing.</li>
                        <li>You may want to include preprocessor parameters in the parameter lookup.</li>
                    </ul>
                    <p class="mb-1"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html" class="decoration-none">ColumnTransformer</a> helps perform different transformations for different columns of data, in a secure Pipeline from data leaks and which can be parametric. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html" class="decoration-none">ColumnTransformer</a> works on arrays, sparse matrices, and pandas DataFrames . For each column, different transformations can be applied, such as preprocessing or extraction methods certain features</p>
                    <p class="mb-1">The <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html" class="decoration-none">Pipelines</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html" class="decoration-none">ColumnTransformer</a> flows used in the model are as follows:</p>
                    <div class="mb-2">
                        <div style="color: #FF69B4;">{model.estimator}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def predict_upload_file(model, df):
    st.warning(f'Uploaded files must match the following column order: **{list(df.columns)}**')
    uploaded_file = st.file_uploader("Choose a file (only supports csv file extension or excel file)")
    try:
        if uploaded_file:
            name = uploaded_file.name
            df_data = pd.read_csv(uploaded_file) if name.split('.')[-1]=='csv' else pd.read_excel(uploaded_file)
            df_data.columns = df.columns
            predict = pd.DataFrame(model.predict(df_data), columns=['Prediction'])
            df_predict = predict.join(df_data)
            csv = convert_df(df_predict)
            st.download_button(label='Download data prediction as csv', data=csv, file_name=f"{name.split('.')[0]} with prediction.csv", mime='text/csv')
            st.write(df_predict)
    except:
        st.warning('Uploaded file extension must be csv or excel file')

def predict_manually(model, df):
    col1, col2, col3 = st.columns(3)
    with col1:
        CRIM = st.number_input('CRIM')
    with col2:
        ZN = st.number_input('ZN')
    with col3:
        INDUS = st.number_input('INDUS')

    col1, col2, col3 = st.columns(3)
    with col1:
        CHAS = st.selectbox('CHAS', df.CHAS.unique())
    with col2:
        NOX = st.number_input('NOX')
    with col3:
        RM = st.number_input('RM')

    col1, col2, col3 = st.columns(3)
    with col1:
        AGE = st.number_input('AGE')
    with col2:
        DIS = st.number_input('DIS')
    with col3:
        RAD = st.selectbox('RAD', df.RAD.unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        TAX = st.number_input('TAX')
    with col2:
        PTRATIO = st.number_input('PTRATIO')
    with col3:
        B = st.number_input('B')

    LSTAT = st.number_input('LSTAT')

    button = st.button('Predict')
    if button:
        df_data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]], columns=df.columns[:-1])
        st.info(f"Prediction is: {round(float(model.predict(df_data)), 4)}")