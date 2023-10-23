import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


st.set_page_config(
    page_title="Airline Customer Satisfaction Dashboard",
    page_icon="✅",
    layout="wide",
)

# dashboard title
st.title("Airline Customer Satisfaction Dashboard")
st.write('---')

# Define age groups or bins
age_bins = [0, 18, 30, 40, 50, 60, 100]
age_labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

# Preprocessing Data
column_to_drop=['Unnamed: 0', 'id']
train_df = train_df.drop(column_to_drop, axis=1)
test_df = test_df.drop(column_to_drop, axis=1)

# Full Dataset (Train+Test)
df=pd.concat([train_df,test_df])

# Encoding
obj_data=["Gender","Customer Type","Type of Travel","Class",]
encoder=LabelEncoder()
for col in obj_data:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])
    print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
    
# # Create a correlation heatmap
# st.subheader('Correlation Heatmap')
# correlation_matrix = train_df.drop(['satisfaction'],axis=1).corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
# st.pyplot(plt)

st.subheader('Correlation Matrix of Factors that Contribute to Customer Satisfaction')
correlation_matrix = train_df.drop(['satisfaction'],axis=1).corr()
fig3 = px.imshow(correlation_matrix)
fig3.update_layout(width=800, height=800)  # Adjust the width and height as needed
st.plotly_chart(fig3)

# Create a new column to store age groups
train_df['Age Group'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels)

# Group the data by age groups
age_groups = train_df.groupby('Age Group')

st.write('---')
st.subheader('Satisfaction Distribution for different Age Groups')

selected_age_group = st.selectbox('Select Age Group', age_labels)

# Define a color theme suitable for a dark theme
colors_dark_theme = ['#AB47BC', '#C69FCD', '#6C797F']
satisfaction_colors = {
    'satisfied': '#C69FCD',
    'neutral or dissatisfied': 'purple',
}


satisfaction_counts = train_df['satisfaction'].value_counts().reset_index()
satisfaction_counts.columns = ['Satisfaction', 'Count']

if selected_age_group:
    group_data = age_groups.get_group(selected_age_group)
    satisfaction_counts_by_age_group = group_data['satisfaction'].value_counts().reset_index()
    satisfaction_counts_by_age_group.columns = ['Satisfaction', 'Count']
    
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
            st.markdown("#### Selected Age Group")
            fig1 = px.pie(
            satisfaction_counts_by_age_group,
            names='Satisfaction',
            values='Count',
            title=f'Satisfaction Distribution for Age Group: {selected_age_group}',
            color='Satisfaction',
            color_discrete_map=satisfaction_colors,
        )
        # Display the pie chart using st.plotly_chart()
            st.plotly_chart(fig1)
        
    with fig_col2:
            st.markdown("#### Compare With")
            fig2 = px.pie(
            satisfaction_counts,
            names='Satisfaction',
            values='Count',
            title='Satisfaction Distribution for All Ages',
            color='Satisfaction',  # Use 'color' to specify colors
            color_discrete_map=satisfaction_colors,  # Map colors to 'Satisfaction' types
        )
            st.write(fig2)


st.write('---')
st.subheader('Satisfaction Distributions for different  Customer Segments')

# Generate bar plots for the selected features
label_data = ["Gender", "Customer Type", "Type of Travel", "Class"]

# Select the feature for which you want to create a bar plot
selected_feature = st.selectbox('Select a Feature', label_data)


if selected_feature:
    # Group the data by satisfaction and the selected feature
    counts = df.groupby(['satisfaction', selected_feature]).size()
    counts = counts.fillna(0).reset_index()

    # Create a Plotly Express bar chart
    fig_bar = px.bar(counts, x='satisfaction', color=selected_feature ,y= 0, barmode="group",
                 title=f'Bar plot of {selected_feature} by satisfaction',
                 color_discrete_sequence=colors_dark_theme)
    fig_bar.update_xaxes(title_text='Satisfaction')
    fig_bar.update_yaxes(title_text='Count')

    # Display the bar chart using st.plotly_chart()
    st.plotly_chart(fig_bar)
    st.dataframe(df.groupby(['satisfaction', selected_feature]).size().unstack())


st.write('---')
    
# load model
loaded_model = pickle.load(open("random_forest.pickle", "rb"))
imputer = pickle.load(open("imputer.pickle", "rb"))
scaler = pickle.load(open("scaler.pickle", "rb"))

X_test = test_df.drop('satisfaction', axis=1)  
y_test = test_df['satisfaction']

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# you can use loaded model to compute predictions
y_predicted = loaded_model.predict(X_test_scaled)
accuracy=loaded_model.score(X_test_scaled, y_test)

# Display Feature Importance
st.subheader('Contributing Factors for Satisfaction')
st.write('Identifying the most important factors that contribute to customer satisfaction is one of the key objectives of this project. This barchart visualize the level of importance of each feature for satisfaction classification using the trained Satisfaction prediction model.')

st.write(f'Prediction Accuracy of the trained model: {accuracy*100} %')

importances = loaded_model.feature_importances_

# Create a DataFrame to store feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# Print the DataFrame
#print(feature_importance_df)

# Create a bar chart to visualize feature importances
fig5= px.bar(
    feature_importance_df,
    x='Importance',
    y='Feature',
    orientation='h',  # Horizontal bar chart
    title='Feature Importances'
)
# Display the plot using st.plotly_chart()
st.plotly_chart(fig5)


st.write('---')

st.subheader('Impact of Enhancing the Quality of Different Services/ Features')
st.write('Users of the dashbord can simulate the impact of enhancing the quality of specific services or features. By selecting a service or feature to enhance, users can observe how this change affects predicted satisfaction levels.')

select_feature = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                 'On-board service', 'Leg room service', 'Baggage handling', 'Check-in service',
                 'Inflight service', 'Cleanliness']

# Select a feature using a selectbox
selected_feature = st.selectbox('Select a Services/Feature to be enhanced:', select_feature)
st.write('Selected service/feature values incremented by one for values less than 5.')

temp_X=X_test

if st.button('Further  Increase the Service Quality'):
    if selected_feature in X_test.columns:
        temp_X[selected_feature] = temp_X[selected_feature].apply(lambda x: x + 1 if x < 5 else x)


# Increment the selected feature by one if its value is less than 5
if selected_feature in X_test.columns:
    temp_X[selected_feature] = X_test[selected_feature].apply(lambda x: x + 1 if x < 5 else x)



prediction_counts=pd.Series(y_predicted).value_counts().reset_index()
prediction_counts.columns = ['Satisfaction', 'Count']

temp_X_imputed = imputer.transform(temp_X)
temp_X_scaled = scaler.transform(temp_X_imputed)
y_predicted_temp = loaded_model.predict(temp_X_scaled)
prediction_counts_temp=pd.Series(y_predicted_temp).value_counts().reset_index()
prediction_counts_temp.columns = ['Satisfaction', 'Count']


fig_col3, fig_col4 = st.columns(2)
with fig_col3:
        st.markdown("### Current Satisfaction")
        fig3 = px.pie(
        prediction_counts,
        names='Satisfaction',
        values='Count',
        title='Current satisfaction level predicted on test set',
        color='Satisfaction',  # Use 'color' to specify colors
        color_discrete_map=satisfaction_colors,  # Map colors to 'Satisfaction' types
    )
        st.write(fig3)
        
with fig_col4:
        st.markdown("### After Service Enhancement")
        fig4 = px.pie(
        prediction_counts_temp,
        names='Satisfaction',
        values='Count',
        title=f'Predicted Satisfaction Distribution after enhancing service/feature: {selected_feature}',
        color='Satisfaction',  # Use 'color' to specify colors
        color_discrete_map=satisfaction_colors,  # Map colors to 'Satisfaction' types
    )
        st.write(fig4)

st.write('---')
st.subheader('New Dataset of Customer Feedback Survey')
st.write('Here I have included a test dataset that is not used for training and evaluation of the ML model. That means it is not seen by the model during training and evaluation. Airline can use a new customer feedback dataset for this purpose.')
st.dataframe(pd.read_csv('test.csv'))