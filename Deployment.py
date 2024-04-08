import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title('Model Deployment: Linear Regression')
st.sidebar.header('User Input Parameters')

def user_input_features():
    avg_session =st.sidebar.text_input('Avg Session Length')
    app_time =st.sidebar.text_input('Time on App')
    web_time =st.sidebar.text_input('Time on Website')
    mem_len =st.sidebar.text_input('Length of Membership')
    try:
        avg_session = float(avg_session)
        app_time = float(app_time)
        web_time = float( web_time)
        mem_len = float(mem_len)
    except ValueError:
        st.error('Please enter numeric values for features.')
        return
    data = {'Avg Session Length': avg_session,
            'Time on App': app_time,
             'Time on Website': web_time,
            'Length of Membership': mem_len}
    features = pd.DataFrame(data,index = [0])
    return features
df =user_input_features()
st.subheader('User Input Parameters')
st.write(df) 
# reading data
data = pd.read_csv('data.csv')
# spliting features and target variable
x = data.iloc[: , : 4]
y = data.iloc[: , 4]
#split train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=42)
# model building
model = LinearRegression()
model.fit(x_train, y_train)
#prediction
prediction = model.predict(df)
#output
st.subheader('Prediction Result')
st.write(prediction)
# feature importance 
important_feature = None
coeff = list(model.coef_)
#finding the best festure
for i,j in zip(coeff, data.columns) :
    max_value = 0
    if i > max_value:
        max_value = i
    if i == max(coeff):
        important_feature = j
#output
st.subheader('Most Important Feature or Variable')
st.write(important_feature)
 
        

