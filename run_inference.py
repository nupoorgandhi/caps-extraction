
from ontology_module import Policy, PolicyList
import os
from openai import OpenAI
import csv

def infer(client, model, messages):
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages= [
            {"role": "system", "content": messages['system_message']}
            ],
        response_format=messages['response_format'],
        
    )

    response = completion.choices[0].message.parsed
    print('usage', completion.usage)

    return response.policy_list

def system_message(html_content):

    return f"""
    Extract a series of policies from the text.
    -- The output should contain only extracted spans that appear in the text. Do not modify the spans.
    -- There may be an number of policies in the text
    -- The policies must be classified as either emissions reduction or adaptation policies.

    Text: 
    {html_content}
    """

def response_format():
    return PolicyList



def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flatten a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # If the list contains complex objects, flatten each item
            if all(isinstance(item, dict) for item in v):
                for i, sub_item in enumerate(v):
                    items.extend(flatten_dict(sub_item, f"{new_key}_{i}", sep=sep).items())
            else:
                items.append((new_key, ", ".join(map(str, v)) if v else None))
        else:
            items.append((new_key, v))
    return dict(items)

def policy_to_nested_dict(policy):
    """
    Convert a Policy object to a nested dictionary dynamically.

    """
    def to_dict(obj):
        if hasattr(obj, "__dict__"):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        else:
            return obj

    return to_dict(policy)

def save_policies_to_csv(policy_lists, fnames, output_path):
    """
    Save policies to a CSV with nested dictionaries written as rows.

    """
    all_rows = []
    all_keys = set()

    # Convert each policy to a nested dictionary and collect keys
    for policy_list, fname in zip(policy_lists, fnames):
        for policy in policy_list:
            policy_dict = flatten_dict(policy_to_nested_dict(policy))
            policy_dict['filename'] = fname
            all_rows.append(policy_dict)
            all_keys.update(policy_dict.keys())

    # Write to CSV
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Policies saved to {output_path}")




def debug():
    client = OpenAI()
    model='gpt-4o-2024-08-06'
    dir_ = './data/policy_segments_positive'
    output_file = "./data/output_policies.csv"

    all_policies = []
    fnames = []
    for fname in os.listdir(dir_)[:100]:
        if fname.endswith('.txt'):
            continue
        with open(os.path.join(dir_, fname), 'r') as file:
            html_content = file.read()
        print('read', fname)
        try:
            policy_list_ = infer(client, model, messages={'system_message':system_message(html_content),
                                        'response_format':response_format()})
            all_policies.append(policy_list_)
            fnames.append(fname)
        except:
            continue

    save_policies_to_csv(all_policies, fnames,  output_file)




        
        
        


