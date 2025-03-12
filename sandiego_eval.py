"""
Code adapted from Sireesh's repo at https://github.com/gsireesh/ddd-benchmark/blob/main/evaluate_and_report.py
"""


import json
from difflib import SequenceMatcher
import fire
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import math


NUMERICAL_COLUMNS = []
TEXT_COLUMNS = ['sector',
       'target_or_performance_metric', 'ghg_reduction_2020',
       'ghg_reduction_2025', 'ghg_reduction_2030', 'ghg_reduction_2035',
       'ghg_reduction_2040', 'ghg_reduction_2050',
       'responsibility_or_implementation_details',
       'responsibility_or_implementation', 'already_projected_amount',
       'quantification_of_ghg_emissions_reductions', 'relative_cost',
       'costs_and_benefits_private', 'costs_and_benefits_public', 'funding',
       'timeframe', 'co_benefits', 'supporting_activities', 'target_year',
        'policy_description']
TEXT_COLUMNS_POLICY = ['policy_description']
ANNOTATED_LOCATIONS = [
    "doc"  
]

NUMERICAL_THRESHOLD = 5

def get_comparison_columns(data_df):
    # return TEXT_COLUMNS
    return TEXT_COLUMNS_POLICY
    

def only_textual(data):
    """Get only textual "columns" from  either a row or series, and normalize the text."""
    if data.empty:
        return data
    if isinstance(data, pd.Series):
        filtered = data[[column for column in data.index if column in TEXT_COLUMNS_POLICY]]
        normalized = filtered.str.lower().str.replace("\W+", "_", regex=True)
    else:
        filtered = data[[column for column in data if column in TEXT_COLUMNS_POLICY]]
        normalized = filtered.apply(
            lambda x: (
                x.str.lower().str.replace("\W+", "_", regex=True) if not x.isnull().any() else x
            )
        )
    return normalized


def get_row_match_score(gt_row, pred_row):
    """Compute a match score between a ground truth row and prediction row
        Sum the similarity score between each of the fields
    """
   
    def overlap(x1, x2):
        if type(x1) != str or type(x2) != str:
            return 0
        else:
            return SequenceMatcher(None, str(x2), str(x1)).ratio() 
    text_score = 0
    for gt, pred in list(zip(only_textual(gt_row).values, only_textual(pred_row).values)):
        text_score += overlap(gt, pred)

    return text_score



def get_alignment_scores(gt_df, pred_df, comparison_columns):
    """Get a matrix of alignment scores between ground truth and predicted rows."""
    alignment_matrix = np.zeros((len(gt_df), len(pred_df)))

    for i, (gt_index, data_row) in enumerate(gt_df[comparison_columns].iterrows()):
        best_match_idx = None
        best_match_score = -float('inf')

        for j, (pred_index, pred_row) in enumerate(pred_df[comparison_columns].iterrows()):
            score = get_row_match_score(data_row, pred_row)
            alignment_matrix[i, j] = score

            if score > best_match_score:
                best_match_score = score
                best_match_idx = pred_index

        # if best_match_idx is not None:
        #     print(f"\nGT row {gt_index} best matched with Pred row {best_match_idx}, Score: {best_match_score}")
        #     print(gt_df.loc[[gt_index]]['policy_description'].values[0], pred_df.loc[[best_match_idx]]['policy_description'].values[0])
        #     # breakpoint()
    return alignment_matrix


def align_predictions(pred_df, alignment_matrix):
    """Compute a heuristic alignment between ground truth and predicted rows.

    This function computes a heuristic match score between a predicted row and all ground truth rows
    and greedily assigns each prediction to a ground truth row based on highest score. In case of
    ties, it defaults to sequential alignment.
    """
    candidate_columns = list(range(len(pred_df)))
    assigned_order = []
    for row in alignment_matrix:
        if len(candidate_columns) == 0:
            break

        # sorting trick taken from https://stackoverflow.com/questions/64238462/numpy-descending-stable-arg-sort-of-arrays-of-any-dtype
        # this produces a descending sort that's still stable; otherwise, the initial element order is reversed. This makes the
        # default heuristic for assigning data alignment in the case of a tie the next data.
        ranked_columns = len(row) - 1 - np.argsort(row[::-1], kind="stable")[::-1]

        i = 0
        top_candidate_column = ranked_columns[0]

        while top_candidate_column not in candidate_columns:
            i += 1
            top_candidate_column = ranked_columns[i]
        assigned_order.append(top_candidate_column)

        candidate_columns.remove(top_candidate_column)

    unaligned_rows = pred_df.iloc[candidate_columns] if len(candidate_columns) != 0 else None
    
    aligned_df = pred_df.iloc[assigned_order]

    if (rows_to_add := (alignment_matrix.shape[0] - len(aligned_df))) > 0:
        none_df = pd.DataFrame({col: [None] * rows_to_add for col in aligned_df})
        aligned_df = pd.concat((aligned_df, none_df))

    return aligned_df, unaligned_rows


def compute_aligned_df_f1(gt_df, aligned_rows, unaligned_rows, present_columns):
    """Compute F1 score for a single paper.

    gt_df and aligned_rows should be the same shape, and unaligned rows are additional rows that
    have been predicted that have no corresponding ground truth row. Present columns are columns
    that have a value in ground truth, i.e. can be found within the page context.
    """
    absent_columns = [
        column
        for column in gt_df
        if column not in present_columns and column in NUMERICAL_COLUMNS + TEXT_COLUMNS
    ]
    tp_numeric, fp_numeric, tn_numeric, fn_numeric = 0, 0, 0, 0
    tp_text, fp_text, tn_text, fn_text = 0, 0, 0, 0
    fp_extra_rows = 0

    source_metrics = {
        (mtype, loc): 0 for mtype in ["tp", "fp", "tn", "fn"] for loc in ANNOTATED_LOCATIONS
    }



    ## METRICS FOR TEXT DATA
    def overlap(x1, x2):
        if type(x1) != str or type(x2) != str:
            return False
        return SequenceMatcher(None, x2, x1).ratio() > .6 

    # breakpoint()
    # aligned_rows = aligned_rows.sort_index()
    # For all present data, compute true and false positives, and false negatives
    for column in set(TEXT_COLUMNS).intersection(set(present_columns)):
        
        location = 'doc'
        new_tp = 0
        new_fp = 0
        new_fn = 0
        for gt, pred in list(zip(list(gt_df[column].values), list(aligned_rows[column].values))):
            if overlap(str(gt), str(pred)):
                new_tp += 1
                breakpoint()
        tp_text += new_tp

        num_true_gold = len(gt_df['policy_id'].unique())
        fn_text = len(gt_df[~gt_df[column].isnull().values & aligned_rows[column].isnull().values]['policy_id'].unique())
        fn_text += new_fn

        fp_text += len(gt_df[~gt_df[column].isnull().values]['policy_id'].unique()) - (new_tp + new_fn)

        if fp_text < 0:
            breakpoint()
        


    ## CALCULATING P, R, F1
    tp = tp_numeric + tp_text
    fn = fn_numeric + fn_text
    fp = fp_numeric + fp_text + fp_extra_rows
    tn = tn_numeric + tn_text
    print(f"tp:{tp}, fn:{fn}, fp:{fp}, tn:{tn}")
    # breakpoint()
    if tp == 0:
        precision, recall, f1 = 0,0,0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    print(f"p:{precision}, r:{recall}")
    return {
        "tp_numeric": tp_numeric,
        "tp_text": tp_text,
        "tp": tp,
        "fp_numeric": fp_numeric,
        "fp_text": fp_text,
        "fp": fp,
        "fn_numeric": fn_numeric,
        "fn_text": fn_text,
        "fn": fn,
        "tn_numeric": tn_numeric,
        "tn_text": tn_text,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "location_metrics": source_metrics,
    }





def evaluate_predictions(gt_df, pred_df):

    if "id" not in pred_df.columns:
        print('id missing', pred_df.columns)
        # pred_df["doi"] = pred_df["source"].str.replace(".png|.xml|.html", "").str.replace("_", "/")

    missing_columns = [
        column for column in NUMERICAL_COLUMNS + TEXT_COLUMNS if column not in pred_df.columns
    ]
    if missing_columns:
        raise AssertionError(f"Predictions dataframe missing required columns: {missing_columns}")

    for column in NUMERICAL_COLUMNS:
        pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")

    print(f"Predicted papers: {len(pred_df['id'].unique())}")
    results_dict = {}
    # group by jurisdiction
    for doi in tqdm(gt_df["id"].unique()):
        ddf = gt_df[gt_df["id"] == doi] # gold jurisidction data
        pdf = pred_df[pred_df["id"] == doi][NUMERICAL_COLUMNS + TEXT_COLUMNS] # pred jurisdication data
        # breakpoint()
        if pdf.empty:
            print(f"DOI {doi} not found in predictions. Skipping.")
            continue
            

        comparison_columns = get_comparison_columns(ddf)
        alignment_matrix = get_alignment_scores(ddf, pdf, comparison_columns)
        aligned_df, unaligned_df = align_predictions(pdf, alignment_matrix)

        comparison_columns = TEXT_COLUMNS
        result = compute_aligned_df_f1(ddf, aligned_df, unaligned_df, comparison_columns)
        results_dict[doi] = result
    
    results_df = pd.DataFrame(results_dict).T
    totals = results_df[["tp", "fp", "tn", "fn"]].sum()
    precision = totals["tp"] / (totals["tp"] + totals["fp"])
    recall = totals["tp"] / (totals["tp"] + totals["fn"])
    f1 = (2 * precision * recall) / (precision + recall)
    print({
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_predictions_wrapper(
    predictions_path: str,
    ground_truth_path="data/zeolite_data_location_annotated.csv",
) -> None:
    """Evaluate predictions against annotations, and provide precision, recall and f1.

    This script calculates and reports both overall metrics and metrics per-location of the data.

    :param predictions_path: Path to CSV containining predictions. Script will error if there are
    expected columns missing. Expected columns are all columns found in NUMERIC_COLUMNS and
    TEXT_COLUMNS in this file.
    :param ground_truth_path: The path to the ground truth CSV. Should not need to be updated.
    :return:
    """
    gt_df = pd.read_csv(ground_truth_path)
    pred_df = pd.read_csv(predictions_path)

    results = evaluate_predictions(gt_df, pred_df)

    json_results = json.dumps(results, indent=4)

    with open(predictions_path.replace(".csv", "_results.json"), "w") as f:
        f.write(json_results)

    print(json_results)


if __name__ == "__main__":
    fire.Fire(evaluate_predictions_wrapper)