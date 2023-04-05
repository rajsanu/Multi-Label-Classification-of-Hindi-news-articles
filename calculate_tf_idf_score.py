import pandas as pd
from function import punctuations, score, load_stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


# Load stopwords from a csv file and store them in a list
stopwords = load_stopwords("stopwords.csv")

# add punctuations in the list of stopwords
stopwords.extend(punctuations)

# get dataframe for the dataset
try:
    dataset = pd.read_excel("preprocessed_data_new.xlsx")
except FileNotFoundError as e:
    raise Exception(e)

# get column names(17 out of which latter 14 are class labels)
cols = dataset.columns.ravel()

# reading features file and storing dataframes for each label in a list
label_features = []
for i in range(3, 17):
    sheet = cols[i]
    feature = pd.read_excel("features.xlsx", sheet_name=sheet)
    label_features.append(feature)

# Applying score() function on each article to get a dataframe which contains 31 columns (17 original columns of raw dataset + 17 new columns containing tf-idf scores for each label) and download it as excel file.

# Define column names for (to be) newly created columns
columns = []
for i in range(3, 17):
    col_name = "score_" + cols[i]
    columns.append(col_name)

# Define a list consisting of 14 empty sub-lists which will store tf-idf scores for labels
result_list = []
for i in range(14):
    a = []
    result_list.append(a)

# apply score function by iterating over each article and store tf-idf scores for each article in previous defined lists
j = 0
for index, row in dataset.iterrows():
    result = score(row['desc'], label_features=label_features, cols=cols, stopwords=stopwords)
    for i in range(14):
        result_list[i].append(result[i])
    j += 1
    print(j) # just to track progress while program executes

# initialise the tf-idf_score columns with lists containing values for each label(for every article)
for i in range(14):
    dataset[columns[i]] = result_list[i]

# convert dataframe to an excel file
dataset.to_excel('tf_idf_score.xlsx', index=False)
