

import multiprocessing
import pandas as pd
import argparse
import numpy as np
from nonconformist.icp import IcpClassifier
from nonconformist.nc import MarginErrFunc, ClassifierNc
from nonconformist.base import ClassifierAdapter
import pickle
from rdkit import Chem
from rdkit.Chem import QED
from molskill.scorer import MolSkillScorer
import requests
import os
from src.data_processing import determine_value, compute_average, mfp

# TODO move functions to a different script, clean this one up
# TODO comment functions
# TODO fix paths
# TODO clean the main function, too many things in there

# Define the model function
def run_model(args):
    range_nruns, input_mfp, loaded_models, loaded_lists = args
    input_df = pd.DataFrame({})
    columns_to_keep= []
    for i in range_nruns:
        if i < 100:
            for z in range(10):
                loaded_model = loaded_models['RFC_' + str(i) + '_' + str(z)]

                model = IcpClassifier(ClassifierNc(ClassifierAdapter(loaded_model), MarginErrFunc()), condition=lambda x: x[1])
                model.cal_scores = loaded_lists['nonconformis_list_' + str(i) + '_' + str(z)]
                model.classes = np.array([0., 1.])

                predict_pvals = model.predict(input_mfp, significance=None)
                input_df = pd.concat([input_df, pd.DataFrame({'inactive_pval_' + str(i) + '_' + str(z): list(predict_pvals[:, 0])})], axis = 1)
                input_df = pd.concat([input_df, pd.DataFrame({'active_pval_' + str(i) + '_' + str(z): list(predict_pvals[:, 1])})], axis = 1)

            # average the folds p values
            input_df['inactive_pval_' + str(i)] = input_df.apply(compute_average, args = ('inactive_pval_' + str(i) + '_', ), axis=1)
            input_df['active_pval_' + str(i)] = input_df.apply(compute_average, args = ('active_pval_' + str(i) + '_', ), axis=1)
            columns_to_keep.append('inactive_pval_' + str(i))
            columns_to_keep.append('active_pval_' + str(i))

    input_df = input_df[columns_to_keep]

    return input_df

def run_CL(input_df, CL):
    CL = 1 - CL
    columns_to_keep = ['Active_count', 'Inactive_count', 'None_count', 'Both_count']
    
    # Get predictions for particular CL
    for i in range(100):
        input_df['class_' + str(i)] = input_df.apply(determine_value, args = (CL, 'inactive_pval_' + str(i), 'active_pval_' + str(i)), axis=1)

    # Get the columns that start with 'class'
    class_columns = [col for col in input_df.columns if col.startswith('class')]

    values_to_count = ['Active', 'Inactive', 'Both', 'None']

    # Iterate through each value and create a new column with the count
    for value in values_to_count:
        input_df[f'{value}_count'] = input_df[class_columns].apply(lambda row: row.str.count(value).sum(), axis=1)
    
    input_df = input_df[columns_to_keep]
    return input_df

def get_druglike_scores(df, column, input_file):

    cmd = f"python ./src/assignSubstructureFilters.py --data {input_file} --smilesColumn {column} --result ./NIBR.csv"
    os.system(cmd)
    nibr = pd.read_csv('./NIBR.csv', sep = ',')
    os.system("rm NIBR.csv")

    # Obtain NIBR, QED and MolSkill scores for all the actives morphs - to export to excel file
    severity_scores = nibr.SeverityScore.tolist()
    smiles = df[column].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    qeds=[QED.qed(m) for m in mols]
    scorer = MolSkillScorer()
    scores = scorer.score(smiles)
    df['QED'] = qeds
    df['NIBR'] = severity_scores
    df['MolSkill'] = scores
    return df

def predict_pEC50(df, mfp_input):
    # import RFR model
    with open('./models/GR_RFR_model.pkl', 'rb') as file:
        RFR_model = pickle.load(file)

    RFR_prediction = RFR_model.predict(mfp_input)
    df['predicted_pEC50'] = RFR_prediction
    return df

def join_results(args, input_df, results, mfp_input):

    # Merge the results by concatenating them horizontally (axis=1)
    final_results = pd.concat(results, axis=1)
    
    # Create a list of arguments for each pool process
    final_results = run_CL(final_results, args.CL)

    final_results['GR_active'] = final_results['Active_count'].apply(lambda x: 1 if x >= 50 else 0)
    final_results = final_results[['GR_active']]

    if args.other_scores == 1:
        input_df = get_druglike_scores(input_df, args.column, args.input_file)
        input_df = predict_pEC50(input_df, mfp_input)

    input_df = pd.concat([input_df, final_results], axis = 1)
    input_df.to_csv(f"{args.output}", sep = ',', index = False)

def main():

    parser = argparse.ArgumentParser(description="RFC-MCCP model trained with active GR ligands.")

    # Positional argument
    parser.add_argument("input_file", help="Input file to predict, csv format including column names")

    # Optional arguments
    parser.add_argument("--output", "-o", help="Output file name", default="GR_MCCP_output.csv")
    parser.add_argument("--column", "-c", help="Column name containing the SMILES", default="smiles")
    parser.add_argument("--CL", type=float, default=0.9, help="Confidence level, float from 0.01-0.99, not valid results < 0.7")
    parser.add_argument("--other_scores", type=int, choices=[0,1], default=0, help="Predict pEC50 and QED, NIBR and MolSkill scores")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes to run")

    args = parser.parse_args()

    num_processes = args.processes

    # Read data to predict
    input_df = pd.read_csv(args.input_file)
    mfp_input = np.array(mfp(input_df, args.column),dtype="float")

    # Run and save the models

    # Import the model data
    # TODO check if it is available, if not, download it from ZENODO

    # download models pickle file if it isn't there still
    model_path = "./models/Model33.pkl"
    if not os.path.exists(model_path):
        print("Downloading models...")
        url = "https://zenodo.org/record/10051801/Model33.pkl"
        req = requests.get(url, allow_redirects=True)
        with open(model_path, "wb") as model_file:
            model_file.write(req.content)
        print("Successfully downloaded!")

    with open('./models/Model33.pkl', 'rb') as file:
        loaded_models = pickle.load(file)

    with open('./models/M33_nonconformist_lists_including_kfold.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)

    # Create a list of arguments for each pool process
    pool_args = [(list(range(i, i+num_processes)), mfp_input, loaded_models, loaded_lists) for i in range(0, 100, num_processes)]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(run_model, pool_args)

    join_results(args, input_df, results, mfp_input)

    print(f"All finished! {args.output} created.")

if __name__ == "__main__":
    main()