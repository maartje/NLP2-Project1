import os

def evaluate_alignments_testset(model_name, alignments):
    path_answers = 'testing/answers/test.wa.nonullalign'
    path_alignments = f'{model_name}.nonullalign'

    write_naacl_alignments(alignments, path_alignments)
    check_naacl_format(path_alignments)
    evaluate_naacl_alignments(path_answers, path_alignments)

def write_naacl_alignments(alignments, path_alignments):
    with open(path_alignments, mode='wt', encoding='utf-8') as alignments_file:
        for i, aligned_sentence in enumerate(alignments):
            sentence_no = f'{(i + 1):04}'
            alignment_lines = [f'{sentence_no} {pos_L1} {pos_L2}' for pos_L1, pos_L2 in aligned_sentence]
            alignment_string = '\n'.join(alignment_lines)
            alignments_file.write(alignment_string)
            alignments_file.write('\n')
    print(f'File generated: {path_alignments}.')
        
def check_naacl_format(path_alignments):
    os.system(f'testing/eval/wa_check_align.pl {path_alignments}')
    print(f'Look in terminal to check if --{path_alignments}-- is in NAACL format...')

def evaluate_naacl_alignments(path_answers, path_alignments):
    os.system(f'testing/eval/wa_eval_align.pl {path_answers} {path_alignments}')
    print('Look in terminal to see AER scores...')

