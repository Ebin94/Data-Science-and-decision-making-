# import required modules
import numpy as np
import pandas as pd
import os
import zipfile

from datetime import timedelta, datetime
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)


# function to recursively unzip files and folders
def unzip(path, file_name):
    
    # define data path
    data_path = os.path.join(path, file_name)
    
    # initialize the zipfile module to unzip all zipped folders
    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    print('Done extracting')


# function to combine data from multiple CSV files into a single dataframe
def combine(path: str, save_path: str, final_columns: dict, names: dict, signals: list):
    # create empty dataframes to store data for each signal
    acc = pd.DataFrame(columns=final_columns['ACC'])
    eda = pd.DataFrame(columns=final_columns['EDA'])
    hr = pd.DataFrame(columns=final_columns['HR'])
    temp = pd.DataFrame(columns=final_columns['TEMP'])

    # helper function to process each dataframe
    def process_df(df, file):
        start_timestamp = df.iloc[0,0]
        sample_rate = df.iloc[1,0]
        new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
        new_df['id'] =  file[-2:] # get the last two characters of the file name and add them as an 'id' column
        new_df['datetime'] = [(start_timestamp + i/sample_rate) for i in range(len(new_df))] # calculate the datetime column based on start timestamp and sample rate
        return new_df

    # loop through all files in the given path
    for file in os.listdir(path):
        # loop through all subdirectories within the current file
        for sub_file in os.listdir(os.path.join(path, file)):
            # check if the current subdirectory contains a CSV file
            if not sub_file.endswith(".zip"):
                # loop through all CSV files within the current subdirectory
                for signal in os.listdir(os.path.join(path, file, sub_file)):
                    # check if the current CSV file corresponds to one of the desired signals
                    if signal in desired_signals:
                        # read the CSV file into a pandas dataframe
                        df = pd.read_csv(os.path.join(path, file, sub_file, signal), names=names[signal], header=None)
                        if not df.empty: # check if the dataframe is not empty
                            # add the dataframe to the appropriate signal dataframe based on the signal type
                            if signal == 'ACC.csv':
                                acc = pd.concat([acc, process_df(df, file)])             
                            if signal == 'EDA.csv':
                                eda = pd.concat([eda, process_df(df, file)])
                            if signal == 'HR.csv':
                                hr = pd.concat([hr, process_df(df, file)])
                            if signal == 'TEMP.csv':
                                temp = pd.concat([temp, process_df(df, file)])

    # write each signal dataframe to a separate CSV file
    acc.to_csv(os.path.join(save_path, 'combined_acc.csv'), index=False)
    eda.to_csv(os.path.join(save_path, 'combined_eda.csv'), index=False)
    hr.to_csv(os.path.join(save_path, 'combined_hr.csv'), index=False)
    temp.to_csv(os.path.join(save_path, 'combined_temp.csv'), index=False)


def merge(path, save_path):
    # Create the save_path directory if it does not exist
    if path != save_path:
        os.makedirs(save_path, exist_ok=True)

    print("Reading data ...")

    # Read the data files into separate dataframes, specifying 'id' column as string type
    acc = pd.read_csv(os.path.join(path, "combined_acc.csv"), dtype={'id': str})
    eda = pd.read_csv(os.path.join(path, "combined_eda.csv"), dtype={'id': str})
    hr = pd.read_csv(os.path.join(path, "combined_hr.csv"), dtype={'id': str})
    temp = pd.read_csv(os.path.join(path, "combined_temp.csv"), dtype={'id': str})

    # Merge the dataframes on the 'id' and 'datetime' columns, using outer join
    print('Merging Data ...')
    df = acc.merge(eda, on=['id', 'datetime'], how='outer')
    df = df.merge(hr, on=['id', 'datetime'], how='outer')
    df = df.merge(temp, on=['id', 'datetime'], how='outer')

    # Fill in missing values in the merged dataframe using forward fill and back fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Save the merged dataframe to a parquet file in the specified save_path
    print("Saving data ...")
    df.to_parquet(os.path.join(save_path, "merged_data.parquet"), index=False)
    
def convert_to_gmt(survey_df):
    print("Converting ...")
    # Set daylight saving time
    daylight = pd.to_datetime(datetime(2020, 11, 1, 0, 0))

    # Copy the dataframe and adjust the datetime for surveys before daylight saving time
    print('Adjust daylight savings')
    survey_df1 = survey_df[survey_df['End datetime'] <= daylight].copy()
    survey_df1['Start datetime'] = survey_df1['Start datetime'].apply(lambda x: x + timedelta(hours=5))
    survey_df1['End datetime'] = survey_df1['End datetime'].apply(lambda x: x + timedelta(hours=5))

    # Copy the dataframe and adjust the datetime for surveys after daylight saving time
    survey_df2 = survey_df.loc[survey_df['End datetime'] > daylight].copy()
    survey_df2['Start datetime'] = survey_df2['Start datetime'].apply(lambda x: x + timedelta(hours=6))
    survey_df2['End datetime'] = survey_df2['End datetime'].apply(lambda x: x + timedelta(hours=6))

    print('Concatenate dataframes')
    # Concatenate the two dataframes and reset the index
    survey_df = pd.concat([survey_df1, survey_df2], ignore_index=True)
    survey_df.reset_index(drop=True, inplace=True)
    return survey_df

def label_data(df, survey_df):
    # Label Data
    print('Labelling ...')

    # Get unique IDs from the input dataframe
    ids = df['id'].unique()

    new_df_list = []

    # Loop through the unique IDs
    for id_ in ids:
        print(f"Processing ID {id_} ...")

        # Create a new dataframe for each ID
        new_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'id', 'datetime', 'label'])

        # Get the data for the current ID from the input dataframe
        sdf = df[df['id'] == id_].copy()

        # Get the survey data for the current ID from the survey dataframe
        survey_sdf = survey_df[survey_df['ID'] == id_].copy()

        # Print out the number of rows for the current ID in both dataframes
        print(f"Found {len(sdf)} rows for ID {id_}")
        print(f"Found {len(survey_sdf)} survey rows for ID {id_}")

        # Loop through the survey data for the current ID
        for _, survey_row in survey_sdf.iterrows():

            # Get the data from the input dataframe that falls within the survey time range
            ssdf = sdf[(sdf['datetime'] >= survey_row['Start datetime']) & (sdf['datetime'] <= survey_row['End datetime'])].copy()

            # If there is data within the survey time range, label it with the survey stress level
            if not ssdf.empty:
                ssdf['label'] = np.repeat(survey_row['Stress level'], len(ssdf.index))
                new_df = pd.concat([new_df, ssdf], ignore_index=True)

            # Otherwise, print out a message indicating that the survey is missing a label for the given time range
            else:
                print(f"{survey_row['ID']} is missing label {survey_row['Stress level']} at {survey_row['Start datetime']} to {survey_row['End datetime']}")
            
        new_df_list.append(new_df) # add new_df to the list of new dataframes
    new_df = pd.concat(new_df_list, ignore_index=True) # concatenate all new dataframes into one
    print('Saving ...')
    PATH = 'C:/Users/HP/Documents/CE888/complete_data' # set directory path
    new_df.to_csv(os.path.join(PATH, 'merged_data_labeled2.csv'), index=False) # save new_df as csv file
    print('Done')

def train_model(model, X_train, X_test, y_train, y_test, X_valid=None, y_valid=None):
    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, use_best_model=True, eval_set=(X_valid, y_valid), verbose=500)
    else:
        model.fit(X_train, y_train)
        
    model_pred = model.predict(X_test)
    acc = accuracy_score(y_test, model_pred)
    pre = precision_score(y_test, model_pred, average='macro')
    rec = recall_score(y_test, model_pred, average='macro')
    f1 = f1_score(y_test, model_pred, average='macro')

    model_scores = [acc, pre, rec, f1]
    classification = classification_report(y_test, model_pred)
    return model, classification, model_scores

