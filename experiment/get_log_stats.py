from entroclus import utils
import pm4py

import os

logs = ['Helpdesk.xes', 'RTFM.xes', 'BPIC13_incidents.xes', 'BPIC13_closedproblems.xes', 'Hospital_Billing.xes', 'Sepsis.xes', 'BPIC12.xes', 'BPIC15.xes']

def get_vocabulary_size(variantlog):
    acts = []
    for variant in list(variantlog.keys()):
        for element in list(variant):
            acts.append(element)
    voc = set(acts)
    return len(voc)
        
table_data = []

for logname in logs:
    print(logname)
    log = pm4py.read_xes("datasets/" + logname)
    total_number_of_cases = log["case:concept:name"].nunique()
    variant_log = utils.get_variant_log(log)
    num_variants = len(variant_log)
    vocabulary_size = get_vocabulary_size(variant_log)
    average_case_length = sum(
        [len(variant) * variant_log[variant] for variant in list(variant_log.keys())]
    ) / total_number_of_cases
    minimum_case_length = min([len(variant) for variant in list(variant_log.keys())])
    maximum_case_length = max([len(variant) for variant in list(variant_log.keys())])

    table_data.append(
        [
            logname,
            total_number_of_cases,
            num_variants,
            vocabulary_size,
            average_case_length,
            minimum_case_length,
            maximum_case_length,
        ]
    )

# Print LaTeX table
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
print("\\hline")
print(
    "Log Name & Total Cases & Variants & Vocabulary Size & Avg Case Length & Min Case Length & Max Case Length \\\\"
)
print("\\hline")

for row in table_data:
    print(" & ".join(map(str, row)) + " \\\\")
    print("\\hline")

print("\\end{tabular}")
print("\\caption{Log Information}")
print("\\label{tab:log_info}")
print("\\end{table}")


    


