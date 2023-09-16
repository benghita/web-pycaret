import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml_toolbox.AutoML import autoML
from io import StringIO
import pickle
import io
st.set_option('deprecation.showPyplotGlobalUse', False)

# initialization of session variabals
if 'page' not in st.session_state:
    st.session_state.page = 0

if 'changed' not in st.session_state:
    st.session_state.changed = False

def next():
    st.session_state.page = st.session_state.page + 1

def pickle_model(model):
    """Pickle the model inside bytes. In our case, it is the "same" as 
    storing a file, but in RAM.
    """
    f = io.BytesIO()
    pickle.dump(model, f)
    return f


def main():

    if st.session_state.page == 0 : 
    # Page 1 : loading data and displaying its info

        st.header("PyCaret Web")
        st.markdown('Upload your dataset : ')
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
         
            dataframe = pd.read_csv(uploaded_file)
            st.dataframe(dataframe)

            # Capture the output of df.info() in a StringIO buffer
            buffer = StringIO()
            dataframe.info(buf=buffer)
            info_str = buffer.getvalue()

            # Display the DataFrame info in Streamlit
            st.write("DataFrame Info:")
            st.code(info_str, language='text')
            
            # Select the target column (y)
            availble_cols = dataframe.columns.tolist()
            st.selectbox('Select the target column :', availble_cols, key = 'target')
            st.session_state.autom = autoML(dataframe, st.session_state.target)
            st.button("Next", on_click = next)

    if st.session_state.page == 1 :
    # Page 2 : Data preprocessing 
    
        with st.form("data_pre"):
                
            st.header('Data Preprocessing')

            st.dataframe(st.session_state.autom.df)

            # Handle missing data
            st.selectbox('Choose the method of imputation for numerical values :',
            ('mean', 'median', 'most_frequent', 'constant'), key = 'num_impute')

            st.number_input('If constant insert a number :', key = 'number')
            
            st.selectbox('Choose the method of imputation for categorical values :',
            ('most_frequent', 'constant'), key = 'ctg_impute')
            st.text_input('If constant insert a value :', key ='category')


            # delete columns
            availble_cols = st.session_state.autom.df.columns.tolist()
            features = [x for x in availble_cols if x != st.session_state.target]
            
            st.multiselect('Select columns to delete:', features, key = 'col_to_drp')

            st.form_submit_button("Apply changes", on_click = next)

    if st.session_state.page == 2 :
    # Page 3 : Visualization
        
        
        if not st.session_state.changed :
            st.session_state.autom.handle_missing_and_types(
                        st.session_state.num_impute,
                        st.session_state.ctg_impute,
                        st.session_state.number,
                        st.session_state.category,
                        st.session_state.col_to_drp
                    )
            st.session_state.changed = True

        df = st.session_state.autom.df
        st.header("Column Chart Viewer")

        # Iterate through each column
        for column in df.columns:
            st.subheader(f"Column: {column}")
            
            # Check the data type of the column
            if df[column].dtype == 'object':
                # For object columns, plot only a histogram
                plt.figure(figsize=(6, 4))
                sns.histplot(df[column], bins=10)
                plt.title(f"Histogram of {column}")
                st.pyplot()
            else:
                # For numerical columns, show histogram and box plot side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                sns.histplot(df[column], bins=10, ax=ax1)
                ax1.set_title(f"Histogram of {column}")
                
                sns.boxplot(x=df[column], ax=ax2)
                ax2.set_title(f"Box Plot of {column}")
                
                st.pyplot(fig)
        st.button("Start Training", on_click = next )
            
    if st.session_state.page == 3 :
    # Page 4 : show the result and download models

        if st.session_state.autom.task == 'C' : 
            st.header('Train & Evaluate Classification Models')
        elif st.session_state.autom.task == 'R' : 
            st.header('Train & Evaluate Regression Models')
        else : st.warning('Can not specify the task')
        
        if  st.session_state.changed :
            st.session_state.autom.s.setup(st.session_state.autom.df, target = st.session_state.autom.target, n_jobs = -1, normalize = True, session_id=123)
            st.text('Training .... execution will terminate shortly - ')
            st.session_state.best = st.session_state.autom.s.compare_models(turbo = True, errors = 'ignore', include = ['lr', 'dt', 'rf'], budget_time = 2, verbose=False)
            st.session_state.changed = False

        best = st.session_state.best
        st.subheader(f'The Best Model : {best}')

        st.subheader('Leaderboard of the trained models  :')
        metrics = st.session_state.autom.s.pull()
        st.write(metrics)

        st.subheader('Analyze the performance of the best model : ')

        if st.session_state.autom.task == 'R' : 
            st.markdown('Plot Error : ')
            st.session_state.autom.s.plot_model(best, plot = 'error',  display_format = 'streamlit')
            st.markdown('Plot residuals : ')
            st.session_state.autom.s.plot_model(best, plot='residuals', display_format = 'streamlit')

        elif st.session_state.autom.task == 'C' : 
            st.markdown('Area Under the Curve : ')
            st.session_state.autom.s.plot_model(best, plot = 'auc',  display_format = 'streamlit')
            st.markdown('Confusion Matrix : ')
            st.session_state.autom.s.plot_model(best, plot='confusion_matrix', display_format = 'streamlit')

        file_1 = pickle_model(best)
        if st.download_button("Download the best model as .pkl file", data=file_1, file_name="best-model.pkl"
                                ) :
            st.success('Model downloaded successfully!')

            
if __name__ == "__main__":
    main()