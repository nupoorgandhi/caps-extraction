
from run_inference import *
import pandas as pd
from tqdm import tqdm
import numpy as np
from sandiego_eval import evaluate_predictions
from collections import defaultdict
# conda activate llama2


def get_sd_prefixes(df):
    """
    gather jurisdiction, year prefixes
    """
    prefixes = []
    for index, row in df[['Jurisdiction', 'Year']].drop_duplicates().iterrows():
        prefix = f"CA_{row['Jurisdiction'].replace(' of ', '_').replace(' ', '')}_{row['Year']}"

        prefixes.append(prefix)

    return prefixes

def filter_uncovered_rows(prefix_to_content_map, gold_df):
    """
    Filter out gold data that does not appear in the html files
    """
    
    prefixes = []

    for index, row in gold_df.iterrows():
        prefix = f"CA_{row['Jurisdiction'].replace(' of ', '_').replace(' ', '')}_{row['Year']}"
        prefixes.append(prefix)
    
    gold_df['prefix'] = prefixes
    
    filtered_rows = []
    coverage_report = []
    
    for prefix in prefix_to_content_map:
        content_str = ' '.join(prefix_to_content_map[prefix])
        df_proc = gold_df.loc[gold_df['prefix'] == prefix]
        
        # Keep rows where at least one action appears in content
        mask = df_proc[[c for c in gold_df.columns if 'Action' in c]].apply(lambda row: any(row.dropna().astype(str).apply(lambda x: x in content_str)), axis=1)
        covered_count = mask.sum()
        total_count = len(df_proc)
        coverage_report.append(f"{prefix}: {covered_count}/{total_count} covered")
        
        filtered_rows.append(df_proc[mask])
    
    for report in coverage_report:
        print(report)
    
    return pd.concat(filtered_rows) if filtered_rows else pd.DataFrame(columns=gold_df.columns)

def verify_gold_doc_coverage(prefix_to_content_map, gold_df):
    
    prefixes = []

    for index, row in gold_df.iterrows():
        prefix = f"CA_{row['Jurisdiction'].replace(' of ', '_').replace(' ', '')}_{row['Year']}"
        prefixes.append(prefix)
    gold_df['prefix'] = prefixes
    for action_column in [c for c in gold_df.columns if 'Action' in c]:
        for prefix in prefix_to_content_map:
            content_str = ' '.join(prefix_content_map[prefix])
            df_proc =  gold_df.loc[gold_df['prefix'] == prefix].dropna(subset=[action_column])
            total_count = df_proc.shape[0]
            covered_count = df_proc[df_proc[action_column].apply(lambda x: x in content_str)].shape[0]
            print(f"{prefix}, {action_column}, total:{total_count}, covered:{covered_count}")


def load_gold(gold_annotations_excel):
    df = pd.read_excel(gold_annotations_excel)
    df = df.drop(columns=['SD CAPs ID', 'Within Document ID', 'Page Start', 'Page End', 'Chapter Name'])
    df = df.rename({'Responsibility / Implementation':'Responsibility / Implementation Organization',
                    'Adaptation or Mitigation or Other':'Adaptation or Mitigation'})
    for c in df.columns:
        print(f'column {c}, num unique {len(set(df[c].values))}, num nonempty {len(df[c]) - sum(df[c].isna())}')
    
    
    attribute_columns = [c for c in df.columns if 'Action' not in c and 'Jurisdiction' not in c and 'Year' != c]
    df_nonan = df.dropna(subset=attribute_columns, how='all')
    
    new_cols = []
    for c in df_nonan.columns:
        if c not in attribute_columns:
            new_cols.append(c)
        else:        
            new_ac = c.lower().replace(' / ', '-or-').replace(' ', '-').replace('---', '-').replace('--', '-').replace('-', '_')
            if new_ac == 'responsibility/implementation':
                new_ac = 'responsibility/implementation-organization'
            new_cols.append(new_ac)
    df_nonan.columns = new_cols
    return df_nonan



def get_prefix_content_map(prefixes, html_dir):
    """
    just run inference for a small subset of san diego files
    
    """
    
    all_policies = []
    fnames = []
    prefix_content_map = defaultdict(list)
    def search_prefix(prefixes, fname):
        for p in prefixes:
            if fname.startswith(p):
                return p
        return None
    num_files = 0

    for fname in os.listdir(html_dir):
        if fname.endswith('.htm') and any([fname.startswith(p) for p in prefixes]):
            with open(os.path.join(html_dir, fname), 'r') as file:
                html_content = file.read()
            prefix_content_map[search_prefix(prefixes, fname)].append(html_content)
            
            
    for prefix, docs in prefix_content_map.items():
        print(f"{prefix}: doc count {len(docs)}")
    return prefix_content_map

def sandiego_inference(prefixes, html_dir, output_file):
    """
    just run inference for a small subset of san diego files
    
    """
    client = OpenAI()
    model = 'gpt-4o-2024-08-06'
    
    all_policies = []
    fnames = []
    
    num_files = 0
    for fname in tqdm(os.listdir(html_dir)):
        if fname.endswith('.htm') and any([fname.startswith(p) for p in prefixes]):
            with open(os.path.join(dir_, fname), 'r') as file:
                html_content = file.read()            
            
            try:
                policy_list_ = infer(client, model, messages={'system_message':system_message(html_content),
                                            'response_format':response_format()})
                all_policies.append(policy_list_)
                fnames.append(fname)
            except:
                continue
            num_files += 1
            if num_files % 4 == 0:
                print(f'processed {num_files} files')

    save_policies_to_csv(all_policies, fnames,  output_file)
           
                
def select_predictions_by_jurisdiction(gold_df, pred_df):
    """
    create a new row for each level of policy description granularity
    """

    map_ = {}
    for index, row in gold_df[['Jurisdiction', 'Year']].drop_duplicates().iterrows():
        prefix = f"CA_{row['Jurisdiction'].replace(' of ', '_').replace(' ', '')}_{row['Year']}"
        map_[(row['Jurisdiction'], row['Year'])] = prefix

    jurisdictions = []
    
    
    # creaate ids based on jurisdiction, year
    ids = []
    for index, row in pred_df.iterrows():
        id_ = ''
        for (jurisdiction, year), prefix in map_.items():
            
            if row['filename'].startswith(prefix):
                id_ = f"CA_{jurisdiction.replace(' of ', '_').replace(' ', '')}_{year}"
        ids.append(id_)
    pred_df['id'] = ids
   
    ids = []
    for index, row in gold_df.iterrows():
        ids.append(f"CA_{row['Jurisdiction'].replace(' of ', '_').replace(' ', '')}_{row['Year']}")
    gold_df['id'] = ids

    action_columns = [col for col in gold_df.columns if 'Action' in col] # policy description columns

    # Create a new dataframe by melting on the 'Action' columns
    df_gold_melted = pd.melt(gold_df, id_vars=[col for col in gold_df.columns if col not in action_columns],
                        value_vars=action_columns,
                        var_name='Action_Type', 
                        value_name='policy_description',
                        ignore_index=False).dropna(subset='policy_description')
    df_gold_melted['policy_id'] = df_gold_melted.index # policy id represents a single row in original gold df

    return pred_df, df_gold_melted


if __name__ == "__main__":
    gold_df = load_gold(gold_annotations_excel='./data/san-diego-gold.xlsx')
    prefixes = get_sd_prefixes(gold_df)
    prefix_content_map = get_prefix_content_map(prefixes, html_dir='./data/policy_segments_positive')
    gold_df_filtered = filter_uncovered_rows(prefix_content_map, gold_df)
    filepath = './data/sd_output_policies.csv'
    pred_df = pd.read_csv(filepath)
    pred_df, gold_df = select_predictions_by_jurisdiction(gold_df_filtered, pred_df)

    # verify_gold_doc_coverage(prefix_content_map, gold_df)
    # sandiego_inference(prefixes, html_dir='./data/policy_segments_positive', output_file = "./data/sd_output_policies.csv")
    result = evaluate_predictions(gold_df, pred_df)
