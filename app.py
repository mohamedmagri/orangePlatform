from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from orangePlatform.pipeline.prediction import PredictionPipeline
from orangePlatform.config.configuration import ConfigurationManager
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.dates import DateFormatter
import base64
from datetime import datetime 




def load_first_data_from_artifact(artifact_folder):
    # Get a list of all files in the artifact folder
    files = os.listdir(artifact_folder)

    # Filter CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Sort the CSV files to ensure consistent order
    csv_files.sort()

    # Check if there are any CSV files in the folder
    if csv_files:
        # Load the first CSV file
        first_csv_file = csv_files[0]
        data_path = os.path.join(artifact_folder, first_csv_file)
        data = pd.read_csv(data_path)
        return data
    else:
        print("No CSV files found in the artifact folder.")
        return None


artifact_folder_path = 'C:\\Users\\mohamed\\Desktop\\stage pfe Orange\\orangePlatform\\orangePlatform\\artifacts\\data_ingestion'
first_data = load_first_data_from_artifact(artifact_folder_path)


if first_data is not None:
    print("First data loaded successfully.")
    print(first_data.tail())








app = Flask(__name__) # initializing a flask app

# first_data['Time'] = first_data.index

df= first_data[['LTE','Period']]
# dates = pd.date_range(start='2022-03-23', periods=len(df), freq='D')
# df = df.to_frame()
# df.set_index(dates, inplace=True)

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    plot = generate_plot(df)
    return render_template("index.html", plot=plot)


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 



def generate_plot(df):
     
     
#     # Ensure dates are in datetime format
    df['Period'] = pd.to_datetime(df['Period'])

#     # Sort DataFrame by date if it's not already sorted
    df.sort_values('Period', inplace=True)

#     # Generate plot based on data for specified date
    plt.figure(figsize=(8, 4))  # Increase figure width
    plt.plot(df['Period'], df['LTE'])
    plt.xlabel('Date')
    plt.ylabel('Gigabyte')
    plt.title('Daily variation of LTE (4G) volume ')

#     # Set date format and tick frequence
    date_format = DateFormatter('%Y-%m-%d')  # Date format YYYY-MM-DD
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))  # Show up to 6 dates on x-axis

    plt.legend()  # Show legend with input date information

    plt.tight_layout()  # Adjust layout to prevent clipping labels

     # Convert plot to base64 for embedding in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    return f'data:image/png;base64,{plot_data}'



@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
            config = ConfigurationManager()
            data_training_config = config.get_model_trainer_config()
            obj = PredictionPipeline(config=data_training_config)
            dict= obj.predict()
            
            index = pd.date_range(start='2024-03-21', periods=len(dict), freq='D')
            df_pred = pd.DataFrame(list(dict.values()), index=index)
            df_real=df['LTE'].copy()
            df_real = df_real.to_frame()

# Now you can set the index
            index1 = pd.date_range(start='2023-03-23', periods=len(df), freq='D')
            df_real.index = index1
            #df_hat=pd.DataFrame(yhat, index=index, columns=['Values'])
            # Plot the time series data
            plt.figure(figsize=(8, 4))
            #print(type(df_pred))
            #print(df_pred.head(10))
            plt.plot(df_pred, color='red')
            #print(type(df_real))
            #print(df_real.head(10))
            plt.plot(df_real, color='blue')
            plt.title('LTE forecasting for the next 2 Weeks')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True)
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data_pred = base64.b64encode(buffer.read()).decode()
            plt.close()
            dict = {datetime.strptime(key, '%Y-%m-%d'): value for key, value in dict.items()}
            table_data = [{'Date': date.strftime('%Y-%m-%d'), 'Value': value} for date, value in dict.items()]

            return render_template('results.html', plot_data_pred=f'data:image/png;base64,{plot_data_pred}', table_data=table_data)

      


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8081, debug=True)
	app.run(host="0.0.0.0", port = 80)