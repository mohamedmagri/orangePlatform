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
    plt.figure(figsize=(12, 6))  # Increase figure width
    plt.plot(df['Period'], df['LTE'])
    plt.xlabel('Date')
    plt.ylabel('LTE')
    plt.title('Prediction Plot for LTE')

#     # Set date format and tick frequency
    date_format = DateFormatter('%Y-%m-%d')  # Date format YYYY-MM-DD
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))  # Show up to 6 dates on x-axis

#     # Add vertical line and annotation for input date
#     input_date = pd.to_datetime(date)
#     input_value = df[df['Date'] == input_date]['Value'].values
#     if len(input_value) > 0:
#         plt.axvline(x=input_date, color='red', linestyle='--', label=f'Input Date: {date}')
#         plt.annotate(f'Value: {input_value[0]}', xy=(input_date, input_value[0]),
#                      xytext=(input_date, input_value[0] + 0.1), ha='right', color='red')
#     else:
#         print(f'Value for input date {date} not found.')

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
            predict = obj.predict()

            return render_template('results.html', prediction = str(predict))

      


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8081, debug=True)
	app.run(host="0.0.0.0", port = 8081)