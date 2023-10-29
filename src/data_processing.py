from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from rdkit import DataStructs

import numpy as np
import pandas as pd
import math
import warnings
from scipy import stats
from src.scaffolds_generation import make_atomic_scaffold_from_smiles
from scipy.optimize import curve_fit
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def mfp(df, column):
    """
    Calculates Morgan fingerprints for molecules in a DataFrame.

    This function computes Morgan fingerprints for molecules represented as SMILES
    strings in a DataFrame. It uses the RDKit library for molecular fingerprinting.

    Args:
        df (pandas.DataFrame): The DataFrame containing SMILES strings.
        column (str): The column name in the DataFrame containing the SMILES.

    Returns:
        list: A list of RDKit Morgan fingerprints (bit vectors) for each molecule.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"SMILES": ["CCO", "CCC", "C=C"]})
        >>> fingerprints = mfp(df, "SMILES")
        >>> print(fingerprints)
        [... RDKit Morgan fingerprints ...]
    """
    output = []
    smiles = df[column].tolist()
    
    # Loop through the DataFrame index and compute Morgan fingerprints for each molecule
    for smi in smiles:
        try:
            # Convert SMILES to RDKit molecule and calculate the fingerprint
            mol = Chem.MolFromSmiles(smi)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            output.append(fingerprint)
        except Exception as e:
            # Skip if there is an error (e.g., invalid SMILES string)
            print(e)
            print(smi)


    return output

def stra_k_fold(k, idx_train1, idx_train2):
    """
    Performs stratified k-fold cross-validation.

    This function divides the dataset into 'k' folds for performing
    stratified k-fold cross-validation. The function takes two index arrays,
    'idx_train1' and 'idx_train2', corresponding to two sets of data. The 'k'
    parameter determines the number of folds.

    Args:
        k (int): The number of folds for cross-validation.
        idx_train1 (numpy.ndarray): An array containing the indices of the first set of data.
        idx_train2 (numpy.ndarray): An array containing the indices of the second set of data.

    Returns:
        tuple: A tuple containing two lists of indices. The first list contains
        the indices for training data, and the second list contains the indices
        for the data used for calibration.

    Example:
        >>> idx_train1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> idx_train2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        >>> k = 3
        >>> train_indices, cal_indices = stra_k_fold(k, idx_train1, idx_train2)
        >>> print(train_indices)
        [array([ 3,  4,  5, 13, 14, 15,  6,  7,  8, 16, 17, 18]), array([ 0,  1,  2, 10, 11, 12,  6,  7,  8, 16, 17, 18]), array([ 0,  1,  2, 10, 11, 12,  3,  4,  5, 13, 14, 15])]
        >>> print(cal_indices)
        [array([ 0,  1,  2, 10, 11, 12]), array([ 3,  4,  5, 13, 14, 15]), array([ 6,  7,  8, 16, 17, 18])]
    """
    # Initialize lists to store the indices for calibration and training data
    indx_cal = []
    indx_ptrain = []

    # Split the data into 'k' approximately equal chunks for both sets
    chunk_train1 = chunk_indexes(idx_train1, k)
    chunk_train2 = chunk_indexes(idx_train2, k)

    for i in range(k):
        # Create a list of indices for calibration data by combining chunks from both sets
        idex_cal_partial = np.concatenate((chunk_train1[i], chunk_train2[i]))
        indx_cal.append(idex_cal_partial)

        # Create a list of indices for training data by excluding the current fold from calibration
        l = [r for r in range(k) if r != i]
        for p in range(len(l)):
            if p == 0:
                idex_ptrain_partial=np.concatenate((chunk_train1[l[p]], chunk_train2[l[p]]))
            else:
                idex_ptrain_partial=np.concatenate((idex_ptrain_partial, chunk_train1[l[p]], chunk_train2[l[p]]))
        indx_ptrain.append(idex_ptrain_partial)

    return indx_ptrain, indx_cal

def chunk_indexes(seq, num):
    """
    Divide a sequence into approximately equal chunks and return a list of chunked sub-sequences.

    Args:
        seq (sequence): The input sequence to be divided into chunks.
        num (int): The number of approximately equal chunks to create.

    Returns:
        list: A list containing chunked sub-sequences of the input sequence.

    Example:
        >>> seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> num_chunks = 3
        >>> chunked_list = chunk_indexes(seq, num_chunks)
        >>> print(chunked_list)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    """
    avg = len(seq) / float(num)
    return [seq[int(i):int(i + avg)] for i in range(0, len(seq), int(avg))]

def class_one_c(prediction, y, significance): # Not used
    """
    Calculates the rate of singleton predictions of a conformal classification model.

    This function computes the rate of singleton predictions, which are prediction sets
    containing only a single class label, of a classification model that uses the conformal
    prediction framework. The 'prediction' array contains the predicted class probabilities,
    'y' contains the true labels, and 'significance' is a significance level that affects
    the singleton prediction rate.

    Args:
        prediction (numpy.ndarray): An array containing the predicted class probabilities.
        y (numpy.ndarray): An array containing the true labels.
        significance (float): The significance level used for singleton prediction.

    Returns:
        float: The rate of singleton predictions of the conformal classification model.

    Example:
        >>> prediction = np.array([0.8, 0.1, 0.3, 0.9, 0.6])
        >>> y = np.array([1, 0, 1, 0, 1])
        >>> significance = 0.5
        >>> singleton_rate = class_one_c(prediction, y, significance)
        >>> print(singleton_rate)
        0.4
    """
    # Convert the predicted probabilities to binary values based on the significance level
    prediction = prediction > significance

    # Count the number of singleton predictions (prediction sets with only one class label)
    n_singletons = np.sum(1 for _ in filter(lambda x: np.sum(x) == 1, prediction))

    # Calculate the rate of singleton predictions by dividing the count by the total number of instances
    singleton_rate = float(n_singletons) / float(y.size)

    return singleton_rate

def class_n_correct(prediction, y, significance): # Not used
    """
    Calculates the number of correct predictions made by a conformal classification model.

    This function computes the number of correct predictions made by a classification model
    that uses the conformal prediction framework. The 'prediction' array contains the predicted
    class probabilities, 'y' contains the true labels, and 'significance' is the significance
    level used to determine the prediction.

    Args:
        prediction (numpy.ndarray): An array containing the predicted class probabilities.
        y (numpy.ndarray): An array containing the true labels.
        significance (float): The significance level used for prediction.

    Returns:
        int: The number of correct predictions made by the conformal classification model.

    Example:
        >>> prediction = np.array([0.8, 0.1, 0.3, 0.9, 0.6])
        >>> y = np.array([1, 0, 1, 0, 1])
        >>> significance = 0.5
        >>> n_correct = class_n_correct(prediction, y, significance)
        >>> print(n_correct)
        2
    """
    # Convert the predicted probabilities to binary values based on the significance level
    prediction = prediction > significance

    # Get unique labels and map true labels to integers for comparison
    labels, y = np.unique(y, return_inverse=True)

    # Create a boolean array indicating correct predictions
    correct = prediction[np.arange(y.size), y]

    # Count the number of correct predictions
    n_correct = np.sum(correct)

    return n_correct

def class_mean_errors(prediction, y, significance): # Not used
    """
    Calculates the average error rate of a conformal classification model.

    This function computes the average error rate of a classification model
    that uses the conformal prediction framework. The 'prediction' array
    contains the predicted labels, 'y' contains the true labels, and
    'significance' is a significance level that affects the error calculation.

    Args:
        prediction (numpy.ndarray): An array containing the predicted labels from the model.
        y (numpy.ndarray): An array containing the true labels.
        significance (float): The significance level used for error calculation.

    Returns:
        float: The average error rate of the conformal classification model.

    Example:
        >>> prediction = np.array([0, 1, 0, 1, 1])
        >>> y = np.array([1, 1, 0, 0, 1])
        >>> significance = 0.05
        >>> error_rate = class_mean_errors(prediction, y, significance)
        >>> print(error_rate)
        0.4
    """
    # Calculate the number of correct predictions using the class_n_correct function
    n_correct = class_n_correct(prediction, y, significance)

    # Calculate the average error rate by dividing the number of incorrect predictions by the total number of instances
    error_rate = 1 - (float(n_correct) / float(y.size))

    return error_rate

def determine_value(row, CL, inactive_col='inactive_pval', active_col='active_pval'):
    """
    Determine the value for a given row based on specified confidence levels.

    Args:
        row (Series): A row of data containing 'inactive_pval' and 'active_pval' columns.
        CL (float): Confidence level threshold for classification.
        inactive_col (str): The name of the column containing inactive p-values.
        active_col (str): The name of the column containing active p-values.

    Returns:
        str: The classification label for the row ('Both', 'Inactive', 'Active', or 'None').
    """
    if row[inactive_col] >= CL and row[active_col] >= CL:
        return 'Both'
    elif row[inactive_col] >= CL and row[active_col] < CL:
        return 'Inactive'
    elif row[inactive_col] < CL and row[active_col] >= CL:
        return 'Active'
    else:
        return 'None'

def prepared_df_nruns_barplot(df):
    """
    Prepare DataFrame for a bar plot.

    Args:
        df (DataFrame): Input DataFrame containing data.

    Returns:
        DataFrame: A modified DataFrame suitable for bar plotting.
    """
    # Calculate the total sum of values for each 'CL' group
    total_sum = df.groupby('CL').mean()[['Both', 'None', 'Active', 'Inactive']].sum(axis=1)
    
    # Calculate the percentages of each column relative to the total sum
    percentage_columns = ['Both', 'None', 'Active', 'Inactive']
    df.groupby('CL').mean()[percentage_columns] = df.groupby('CL').mean()[percentage_columns].div(total_sum, axis=0) * 100

    # Create a DataFrame with the desired columns
    data = df.groupby('CL').mean()[['None', 'Both', 'Active', 'Inactive']]

    # Rename the index
    data.index = [CL for CL in range(100, 0, -1)]

    # Return the prepared DataFrame
    return data

def frange(start, end=None, inc=1.0):
    """
    A range function that accepts float increments.

    This function generates a list of floating-point numbers starting from 'start'
    and ending before 'end' with a specified increment 'inc'. If 'end' and 'inc'
    are not provided, the function will generate a sequence starting from 0.0
    with an increment of 1.0.

    Args:
        start (float): The starting value of the range or the only value if 'end' is not provided.
        end (float, optional): The ending value of the range. If not provided, it defaults to 'start'.
        inc (float, optional): The increment between consecutive values. If not provided, it defaults to 1.0.

    Returns:
        list: A list containing the generated floating-point numbers.

    Example:
        >>> frange(0.0, 5.0, 0.5)
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        >>> frange(3.0)
        [0.0, 1.0, 2.0, 3.0]
    """
    # If 'end' is not provided, set it to 'start' and update 'start' to 0.0
    if end is None:
        end = start + 0.0
        start = 0.0

    L = []  # Initialize an empty list to store the generated values

    # Generate the range of values based on the provided 'start', 'end', and 'inc'
    while (end - start) * inc > 0:
        L.append(round(start, 2))  # Add the current value to the list
        start += inc  # Increment the value for the next iteration

    return L  # Return the list of generated floating-point numbers

def accuracy_precision(df):
    """
    Compute statistical values from TP/TN/FP/FN matrix (considering Both nor None).

    This function takes a TP/TN/FP/FN matrix represented as a DataFrame 'count1' and
    a list of truth values 'truth'. It calculates various statistical values, such as
    TP, TN, FP, FN, MCC (Matthews correlation coefficient), sensitivity, specificity,
    P (total positive instances), and N (total negative instances).

    Args:
        df (pandas.DataFrame): Dataset with containing the true value and the prediction

    Returns:
        tuple: A tuple containing the calculated statistical values.

    """
    TP = len(df.loc[(df['class'] == 'Active') & (df['activity'] == 1)]) + len(df.loc[(df['class'] == 'Both') & (df['activity'] == 1)])
    TN = len(df.loc[(df['class'] == 'Inactive') & (df['activity'] == 0)]) + len(df.loc[(df['class'] == 'Both') & (df['activity'] == 0)])
    FP = len(df.loc[(df['class'] == 'Active') & (df['activity'] == 0)]) + len(df.loc[(df['class'] == 'Both') & (df['activity'] == 0)])
    FN = len(df.loc[(df['class'] == 'Inactive') & (df['activity'] == 1)]) + len(df.loc[(df['class'] == 'Both') & (df['activity'] == 1)])
    MCC = 0.0
    sensitivity = 0.0
    specificity = 0.0
    P = len(df.loc[(df['activity'] == 1)]) 
    N = len(df.loc[(df['activity'] == 0)])

    # Calculate MCC, sensitivity, and specificity if possible
    if float(math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))) == 0:
        MCC = 0
    else:
        MCC = float(TP * TN - FP * FN) / float(math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    sensitivity = float(TP) / float(P)
    specificity = float(TN) / float(N)
    accuracy = float((TP + TN)/len(df))
    efficiency = (len(df.loc[(df['class'] == 'Active') & (df['activity'] == 1)]) + len(df.loc[(df['class'] == 'Inactive') & (df['activity'] == 0)]))/len(df)
    validity = (len(df.loc[df['class'] == 'Both']) + len(df.loc[(df['class'] == 'Active') & (df['activity'] == 1)]) + len(df.loc[(df['class'] == 'Inactive') & (df['activity'] == 0)]))/len(df)

    return TP, TN, FP, FN, accuracy, efficiency, MCC, sensitivity, specificity, P, N, validity

def compute_average(row, col_name):
    """
    Compute the average of columns in a DataFrame row that match a specified column name pattern.

    Parameters:
        row (pandas.Series): A row from a pandas DataFrame.
        col_name (str): The pattern to match column names for averaging.

    Returns:
        float: The average value of columns matching the specified pattern.
    """
    # Filter columns that match the specified pattern
    avg_columns = row.filter(like = col_name)
    
    # Compute the mean of the filtered columns
    return avg_columns.mean()

def process_test_set(model_name, save=True, filepath=None, path='./results/', kfold=False):
    """
    Process the test set results for a machine learning model.

    Args:
        model_name (str): The name of the machine learning model.
        save (bool): Whether to save the processed results to a CSV file.
        filepath (str): The file path for saving the results (if save=True).
        path (str): The directory path where the raw test results are located.
        kfold (bool): Whether the test results include k-fold cross-validation.

    Returns:
        pd.DataFrame: A DataFrame containing the processed results for different confidence levels (CL).
    """
    # Initialize lists to store performance metrics for different CL values
    TPss, TNss, FPss, FNss, accss, effss, MCCss, senss, spess, Pss, Nss, nones, boths, acts, inacts, CLs, valss, runss = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # Iterate through CL values
    for CL in frange(0, 1, 0.01):  # Assuming frange is a function to generate a range of float values
        num_runs = 100  # Number of runs
        for nrun in range(num_runs):
            # Load the test data
            if kfold:
                df = pd.read_csv(f'{path}{model_name}_test_pvals_run_{nrun}_including_kfolds.csv', sep=',')
                df['inactive_pval'] = df.apply(compute_average, args=('inactive_pval_',), axis=1)
                df['active_pval'] = df.apply(compute_average, args=('active_pval_',), axis=1)
                df = df[['chembl_id', 'activity', 'inactive_pval', 'active_pval']]
            else:
                df = pd.read_csv(f'{path}{model_name}_test_pvals_run_{nrun}.csv', sep=',')

            # Classify data based on CL
            df['class'] = df.apply(determine_value, args=(CL,), axis=1)
            
            # Calculate performance metrics
            TP, TN, FP, FN, accuracy, efficiency, MCC, sensitivity, specificity, P, N, validity = accuracy_precision(df)
            
            # Append metrics to respective lists
            CLs.append(CL)
            runss.append(nrun)
            TPss.append(TP)
            TNss.append(TN)
            FPss.append(FP)
            FNss.append(FN)
            accss.append(accuracy)
            effss.append(efficiency)
            MCCss.append(MCC)
            senss.append(sensitivity)
            spess.append(specificity)
            Pss.append(P)
            Nss.append(N)
            nones.append(len(df.loc[df['class'] == 'None']))
            boths.append(len(df.loc[df['class'] == 'Both']))
            acts.append(len(df.loc[df['class'] == 'Active']))
            inacts.append(len(df.loc[df['class'] == 'Inactive']))
            valss.append(validity)

    # Create a DataFrame to store the processed results
    df = pd.DataFrame({
        'CL': CLs,
        'runs': runss,
        'TP': TPss,
        'TN': TNss,
        'FP': FPss,
        'FN': FNss,
        'accuracy': accss,
        'efficiency': effss,
        'validity': valss,
        'MCC': MCCss,
        'sensitivity': senss,
        'specificity': spess,
        'P': Pss,
        'N': Nss,
        'None': nones,
        'Both': boths,
        'Active': acts,
        'Inactive': inacts
    })

    # Save the processed results to a CSV file if specified
    if save and filepath is None:
        df.to_csv(f'./results/results_processed_{model_name}.csv', sep=',', index=None)
    elif save and filepath is not None:
        df.to_csv(filepath, sep=',', index=None)

    return df

#
def process_results(df, CL, save=True, filepath=None, length_columns_org=2):
    """
    Process the results DataFrame by computing averages and adding classification columns for different CL values.

    Args:
        df (pd.DataFrame): The input DataFrame containing results data.
        CL (float): The confidence level for classification.
        save (bool): Whether to save the processed results to a CSV file.
        filepath (str): The file path for saving the processed results (if save=True).
        length_columns_org (int): The number of original columns to keep.

    Returns:
        pd.DataFrame: A DataFrame containing the processed results with additional classification and count columns.
    """
    # Create a list to keep the original columns
    columns_to_keep = df.columns.tolist()[:length_columns_org]

    # Iterate through different runs
    for i in range(100):
        # Compute averages for inactive and active pval columns and add them to the DataFrame
        df['inactive_pval_' + str(i)] = df.apply(compute_average, args=('inactive_pval_' + str(i) + '_',), axis=1)
        df['active_pval_' + str(i)] = df.apply(compute_average, args=('active_pval_' + str(i) + '_',), axis=1)
        
        # Determine classification for each run
        df['class_' + str(i)] = df.apply(determine_value, args=(CL, 'inactive_pval_' + str(i), 'active_pval_' + str(i)), axis=1)

    # Get columns starting with 'class'
    class_columns = [col for col in df.columns if col.startswith('class')]

    # Define values to count
    values_to_count = ['Active', 'Inactive', 'None', 'Both']

    # Iterate through each value and create new count columns
    for value in values_to_count:
        df[f'{value}_count'] = df[class_columns].apply(lambda row: row.str.count(value).sum(), axis=1)
        columns_to_keep.append(f'{value}_count')

    # Keep only the selected columns
    df = df[columns_to_keep]

    # Save the processed results to a CSV file if specified
    if save and filepath is not None:
        df.to_csv(filepath, sep=',', index=False)

    return df

def hist_similarity(data_input_molpher, data_actives, RML_data=False, RML_input=False, novel=False, sca=False):
    """
    Compute similarity metrics between input and active compounds and generate histograms.

    Args:
        data_input_molpher (pd.DataFrame): DataFrame containing input compound data.
        data_actives (pd.DataFrame): DataFrame containing active compound data.
        RML_data (bool): Flag indicating if data_actives is in RML format.
        RML_input (bool): Flag indicating if data_input_molpher is in RML format.
        novel (bool): Flag indicating if input compounds are novel.
        sca (bool): Flag indicating if atomic scaffold fingerprints should be used.

    Returns:
        list: List of identical GRML IDs.
        list: List of identical ChEMBL IDs.
        list: List of highest Tanimoto coefficients.
        list: List of average Tanimoto coefficients.
    """
    # Extract input SMILES and IDs
    if novel:
        inp_smiles = data_input_molpher.canonical_smiles.tolist()
    else:
        inp_smiles = data_input_molpher.smiles.tolist()
    if novel:
        inp_ids = data_input_molpher.chembl_id.tolist()
    elif not RML_input:
        inp_ids = data_input_molpher.id_in_source.tolist()

    # Generate Morgan fingerprints for input compounds
    if sca:
        inp_smiles_fps = [AllChem.GetMorganFingerprintAsBitVect(make_atomic_scaffold_from_smiles(x), 2, nBits=1024) for x in inp_smiles]
    else:
        inp_smiles_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in inp_smiles]

    # Extract GR90 SMILES and IDs
    GR90_smiles = data_actives.loc[data_actives['Active_count'] >= 50].morph_smiles.tolist()
    if RML_data:
        GR90_ids = data_actives.loc[data_actives['Active_count'] >= 50].id_morph.tolist()
    else:
        GR90_ids = data_actives.loc[data_actives['Active_count'] >= 50].id_morph.tolist()

    # Generate Morgan fingerprints for GR90 compounds
    if sca:
        GR90_smiles_fps = [AllChem.GetMorganFingerprintAsBitVect(make_atomic_scaffold_from_smiles(x), 2, nBits=1024) for x in GR90_smiles]
    else:
        GR90_smiles_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in GR90_smiles]

    highest_TC, avg_TC, identical_GRML, identical_Ch, identical_GRML_smi, identical_Ch_smi = [], [], [], [], [], []

    # Calculate Tanimoto coefficients
    if not RML_input and novel == False and RML_data == False:
        for i, fps in enumerate(GR90_smiles_fps):
            start_id = data_actives.iloc[i].start
            target_id = data_actives.iloc[i].target
            total_compared, total_tca, tc = 0, 0, 0
            for x, fps_inp in enumerate(inp_smiles_fps):
                if (data_input_molpher.iloc[x].id_in_source != start_id) and (data_input_molpher.iloc[x].id_in_source != target_id):
                    total_compared += 1
                    tca = DataStructs.FingerprintSimilarity(fps, fps_inp)
                    total_tca += tca
                    if tca == 1.0:
                        identical_GRML.append(GR90_ids[i])
                        identical_Ch.append(inp_ids[x])
                        identical_GRML_smi.append(GR90_smiles[i])
                        identical_Ch_smi.append(inp_smiles[x])
                    if tca > tc:
                        tc = tca
            highest_TC.append(tc)
            avg_TC.append(float(total_tca / total_compared))

    # Handle other cases (RML input, RML data, novel compounds)
    elif not RML_input and RML_data:
        for i, fps in enumerate(GR90_smiles_fps):
            total_compared, total_tca, tc = 0, 0, 0
            for x, fps_inp in enumerate(inp_smiles_fps):
                total_compared += 1
                tca = DataStructs.FingerprintSimilarity(fps,fps_inp)
                total_tca += tca
                if tca == 1.0:
                    identical_GRML.append(GR90_ids[i])
                    identical_Ch.append(inp_ids[x])
                    identical_GRML_smi.append(GR90_smiles[i])
                    identical_Ch_smi.append(inp_smiles[x])
                if tca > tc:
                    tc = tca
            highest_TC.append(tc)
            avg_TC.append(float(total_tca/total_compared))

    elif RML_input and RML_data:
        for i, fps in enumerate(GR90_smiles_fps):
            start_id = data_actives.iloc[i].id_start_structure
            target_id = data_actives.iloc[i].id_target_structure
            total_compared, total_tca, tc = 0, 0, 0
            for x, fps_inp in enumerate(inp_smiles_fps):
                if (data_input_molpher.iloc[x].id_structure != start_id) and (data_input_molpher.iloc[x].id_structure != target_id):
                    total_compared += 1
                    tca = DataStructs.FingerprintSimilarity(fps,fps_inp)
                    total_tca += tca
                    if tca == 1.0:
                        identical_GRML.append(GR90_ids[i])
                        identical_Ch.append(inp_ids[x])
                        identical_GRML_smi.append(GR90_smiles[i])
                        identical_Ch_smi.append(inp_smiles[x])
                    if tca > tc:
                        tc = tca
            highest_TC.append(tc)
            avg_TC.append(float(total_tca/total_compared))

    elif novel:
        for i, fps in enumerate(GR90_smiles_fps):
            total_compared, total_tca, tc = 0, 0, 0
            for x, fps_inp in enumerate(inp_smiles_fps):
                total_compared += 1
                tca = DataStructs.FingerprintSimilarity(fps,fps_inp)
                total_tca += tca
                if tca == 1.0:
                    identical_GRML.append(GR90_ids[i])
                    identical_Ch.append(inp_ids[x])
                    identical_GRML_smi.append(GR90_smiles[i])
                    identical_Ch_smi.append(inp_smiles[x])
                if tca > tc:
                    tc = tca
            highest_TC.append(tc)
            avg_TC.append(float(total_tca/total_compared))

    elif RML_input:
        for i, fps in enumerate(GR90_smiles_fps):
            start_id = data_actives.iloc[i].start
            target_id = data_actives.iloc[i].target
            total_compared, total_tca, tc = 0, 0, 0
            for x, fps_inp in enumerate(inp_smiles_fps):
                if (data_input_molpher.iloc[x].id_structure != start_id) and (data_input_molpher.iloc[x].id_structure != target_id):
                    total_compared += 1
                    tca = DataStructs.FingerprintSimilarity(fps,fps_inp)
                    total_tca += tca
                    if tca == 1.0:
                        identical_GRML.append(GR90_ids[i])
                        identical_Ch.append(inp_ids[x])
                        identical_GRML_smi.append(GR90_smiles[i])
                        identical_Ch_smi.append(inp_smiles[x])
                    if tca > tc:
                        tc = tca
            highest_TC.append(tc)
            avg_TC.append(float(total_tca/total_compared))
    
    identical_GRML, identical_Ch, _ = same_morph_fil(identical_GRML, identical_Ch, identical_GRML_smi, identical_Ch_smi)
    print(f"Low similarity {sum(i < 0.4 for i in highest_TC)}")
    print(f"Medium similarity {sum((i >= 0.4) and (i < 0.7) for i in highest_TC)}")
    print(f"High similarity {sum(i >= 0.7 for i in highest_TC)}")
    print(f"Identical: {len(identical_Ch)}")
    print(f"Mean Tc: {np.asarray(highest_TC).mean()}")
    print(f"Median Tc: {np.median(np.asarray(highest_TC))}")
    print(f"Mode Tc: {stats.mode(np.asarray(highest_TC))}")
    return identical_GRML, identical_Ch, highest_TC, avg_TC

def KL_divergence_prep(df1, df2, smiles1, smiles2):
    """
    Prepare data and compute similarity metrics between two sets for KL divergence.

    Args:
        df1 (pd.DataFrame): First DataFrame containing data.
        df2 (pd.DataFrame): Second DataFrame containing data.
        smiles1 (str): Column name for SMILES in df1.
        smiles2 (str): Column name for SMILES in df2.

    Returns:
        list: List of highest Tanimoto coefficients between the two sets.
    """
    highest_TC, avg_TC = [], []

    # Generate Morgan fingerprints for compounds in df1 and df2
    fps1 = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in df1[smiles1].tolist()]
    fps2 = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in df2[smiles2].tolist()]

    # Calculate Tanimoto coefficients between compounds in df1 and df2
    for fps in fps1:
        total_compared, total_tca, tc, tca = 0, 0, 0, 0
        for fps_inp in fps2:
            total_compared += 1
            tca = DataStructs.FingerprintSimilarity(fps, fps_inp)
            total_tca += tca
            if tca > tc:
                tc = tca
        highest_TC.append(tc)
        avg_TC.append(float(total_tca / total_compared))

    # Print some statistics
    #print(str(sum(i < 0.4 for i in highest_TC)))
    #print(str(sum(i == 1 for i in highest_TC)))
    #print(np.asarray(highest_TC).mean())

    return highest_TC

def smiles_to_inchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        inchi_key = inchi.MolToInchiKey(mol)
        return inchi_key
    else:
        return None

def same_morph_fil(morph_ids, inputs_id, morphs_smiles, input_smiles):
    """
    Check if the Tc == 1 refer to truly identical compounds
    """    
    # repeated inputs
    non_identical_index = get_second_occurrence_indices(inputs_id)

    # is it really the same compound?
    for i, morph_smi in enumerate(morphs_smiles):
        morph_inchi = smiles_to_inchi(morph_smi)
        input_inchi = smiles_to_inchi(input_smiles[i])
        if morph_inchi != input_inchi:
            non_identical_index.append(i)

    morph_ids_2 = [m_id for i,m_id in enumerate(morph_ids) if i not in non_identical_index]
    inputs_id_2 = [i_id for i,i_id in enumerate(inputs_id) if i not in non_identical_index]

    return morph_ids_2, inputs_id_2, len(non_identical_index)

#
def normalize_and_weight(values):
    """
    Normalize and weight a list of values.

    Args:
        values (list or np.ndarray): List of values to be normalized and weighted.

    Returns:
        np.ndarray: Weighted and normalized values.
    """
    # Normalize the values to be between 0 and 1
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Calculate the sum of normalized values
    sum_normalized = np.sum(normalized_values)

    # Calculate weighted normalized values
    weighted_normalized = normalized_values / sum_normalized

    return weighted_normalized

#
def peak_position(data):
    # Create a histogram
    hist, bins = np.histogram(data, bins=100)

    # Find the bin with the maximum count
    peak_bin = np.argmax(hist)

    # Calculate the position of the peak
    peak_position = (bins[peak_bin] + bins[peak_bin + 1]) / 2

    # Print the position of the peak
    print('Position of the peak:', peak_position)

#
def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function.
    """
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

#
def find_peak(data):
    # Create a histogram
    hist, bins = np.histogram(data, bins=100)

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit a Gaussian curve to the histogram
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[1.0, np.mean(data), np.std(data)])

    # Calculate the position of the peak (mean of the Gaussian)
    peak_position = popt[1]

    # Extract the parameters of the Gaussian curve
    amplitude, mean, stddev = popt

    # Calculate goodness-of-fit metrics
    predicted_values = gaussian(bin_centers, amplitude, mean, stddev)
    r_squared = r2_score(hist, predicted_values)
    mse = mean_squared_error(hist, predicted_values)

    # Print the position of the peak
    print('Position of the peak:', peak_position)

def get_second_occurrence_indices(lst):
    seen = {}  # Dictionary to store the indices of elements
    second_occurrence_indices = []  # List to store indices of second occurrences

    for i, item in enumerate(lst):
        if item in seen:
            if seen[item] is not None:
                second_occurrence_indices.append(seen[item])
            seen[item] = None  # Mark as seen but don't append the first occurrence
        else:
            seen[item] = i

    return second_occurrence_indices

#
def get_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = list(set1.intersection(set2))
    return common_elements