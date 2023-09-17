import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_csv_filepaths(directory_path='/home/modejota/Deep_Var_BCD/results/results_reduced_datasets/', training_type=None):
    """Given a directory path, return a list of all the CSV file paths in that directory and its subdirectories.
    Args:
        directory_path (str): The path to the directory
        training_type (str): The type of training to use. It can be "per_image_training" or "batch_training"
    Returns:
        list: A list of all the CSV file paths in that directory and its subdirectories
    """
    csv_filepaths = []
    for root, _, files in os.walk(directory_path):
        if training_type in root:
            for file in files:
                if file.endswith(".csv"):
                    csv_filepaths.append(os.path.join(root, file))
    
    return csv_filepaths

def get_certain_csv_filepaths(file_termination, directory_path='/home/modejota/Deep_Var_BCD/results_reduced_datasets/', training_type=None):
    """Given a directory path, return a list of all the CSV file paths in that directory and its subdirectories
    Args:
        directory_path (str): The path to the directory
        training_type (str): The type of training to use. It can be "per_image_training" or "batch_training"
    Returns:
        list: A list of all the CSV file paths in that directory and its subdirectories
    """
    csv_filepaths = []
    for root, _, files in os.walk(directory_path):
        if training_type in root:
            for file in files:
                filepath = os.path.abspath(os.path.join(root, file))
                if filepath.endswith(file_termination):
                    csv_filepaths.append(os.path.join(root, file))
    
    return csv_filepaths

def get_model_and_organs_info(csv_filepath, training_type=None):
    """Given a CSV file path, extract the model name and organ info from the file path
    Args:
        csv_filepath (str): The path to the CSV file
    Returns:
        tuple: A tuple containing the model name and organ info
    """
    if training_type == 'batch_training':
        match = re.search(r'(bcdnet_e[1234]|cnet_e2).*?(Breast|Colon|Lung)', csv_filepath)
        if match:
            method = match.group(1)
            organ = match.group(2)
            return method, organ
        else:
            return None, None
    elif training_type == 'per_image_training':
        path_parts = csv_filepath.split(os.sep)
        
        # Extract the desired components from the path
        model_name = path_parts[-4]
        organ_info = path_parts[-2]
        
        return model_name, organ_info
    
def group_csv_files_by_approach(csv_filepaths):
    """Given a list of CSV file paths, group them by approach
    Args:
        csv_filepaths (list): A list of CSV file paths
    Returns:
        dict: A dictionary containing the CSV file paths grouped by approach
    """
    approaches = {}
    for csv_filepath in csv_filepaths:
        path_parts = csv_filepath.split(os.sep)

        approach_name = path_parts[-4]

        if approach_name not in approaches:
            approaches[approach_name] = []
        approaches[approach_name].append(csv_filepath)
    
    return approaches

def generate_graphs_by_approach(indir='/home/modejota/Deep_Var_BCD/results/',
                                outdir='/home/modejota/Deep_Var_BCD/results/graphs/',
                                training_type='per_image_training',
                                metrics_to_use=["psnr", "mse", "ssim"]):
    """Generate one graph for each approach and metric given a directory containing the CSV files with the metrics for a set of images. 
       Meant to be used for individual image training, if done for a big dataset graph could be too big to be useful
    Args:
        indir (str): The path to the directory containing the CSV files
        outdir (str): The path to the directory where the graphs will be saved
        metrics_to_use (list): A list containing the metrics to use. psnr, mse and ssim are the only valid options
    """
    indir += os.sep if indir[-1] != os.sep else ''
    outdir += os.sep if outdir[-1] != os.sep else ''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    csv_files = get_csv_filepaths(indir, training_type=training_type)
    # print(f'Found {len(csv_files)} CSV files.')
    if len(csv_files) == 0:
        return

    method_groups = group_csv_files_by_approach(csv_files)

    for method_name in method_groups:
        print(f'Generating graphs for approach: {method_name}')
        csv_files = method_groups[method_name]
        organ_info = []
        model_name = []
        data_lists = {metric: [] for metric in metrics_to_use}

        csv_files.sort()
        for filename in csv_files:
            filepath = os.path.join(indir, filename)
            model_name_, organ_info_ = get_model_and_organs_info(filepath, training_type=training_type)
            organ_info.append(organ_info_)
            model_name.append(model_name_)

            df = pd.read_csv(filepath)
            for metric in metrics_to_use:
                columns = [f'{metric}_gt', f'{metric}_gt_e', f'{metric}_gt_h']
                for column in columns:
                    if column in df.columns:
                        data = df[column]
                        data_lists[metric].append(data)

        num_rows = (len(organ_info) + 2) // 3

        for metric in metrics_to_use:
            fig, ax = plt.subplots(num_rows, 3, figsize=(10*num_rows, 5 * num_rows))
            fig.suptitle(f'Approach: {method_name} - Metric: {metric.upper()}')

            for row in range(num_rows):
                for col in range(3):
                    idx = row * 3 + col
                    if idx < len(organ_info):
                        organ = organ_info[idx]
                        ax[row, col].set_title('Organ: ' + organ)
                        ax[row, col].set_xlabel('Iterations')
                        ax[row, col].set_ylabel('Value')

                        data_idx = idx * 3
                        ax[row, col].plot(data_lists[metric][data_idx + 1], label=f'{metric}_gt_e', color='steelblue')
                        ax[row, col].plot(data_lists[metric][data_idx + 2], label=f'{metric}_gt_h', color='orange')
                        ax[row, col].plot(data_lists[metric][data_idx], label=f'{metric}_gt', color='green')


                        ax[row, col].legend()

            plt.savefig(os.path.join(outdir, f'{method_name}_{metric.upper()}.png'))
            plt.close()

def generate_graphs_by_image(organ: str, id: int, 
                             indir ='/home/modejota/Deep_Var_BCD/results_reduced_datasets/',
                             outdir='/home/modejota/Deep_Var_BCD/results_reduced_datasets/graphs/',
                             training_type='per_image_training',
                             metrics_to_use=["psnr", "mse", "ssim"]):
    """Generate one graph for each image and metric given a directory containing the CSV files with the metrics for a set of approaches
    Args:
        organ (str): The name of the organ
        id (str): The id of the image
        outdir (str): The path to the directory where the graphs will be saved
    """
    organ = organ.lower().capitalize()
    if id < 0:
        return
    filename = f'{organ}_{id}/metrics.csv' if indir == '/home/modejota/Deep_Var_BCD/results_reduced_datasets/' else f'{organ}_{id}.csv'

    outdir += os.sep if outdir[-1] != os.sep else ''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    csv_files = get_certain_csv_filepaths(filename, directory_path=indir, training_type=training_type)
    csv_files.sort()

    num_files = len(csv_files)
    # print(f'Found {num_files} CSV files for {organ}_{id}.')
    print(f'Generating graphs for {organ}_{id}.')
    if num_files == 0:
        return

    for metric in metrics_to_use:
        num_columns = math.isqrt(num_files)
        while num_files % num_columns != 0:
            num_columns -= 1
        num_rows = num_files // num_columns
        num_rows, num_columns = num_columns, num_rows

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(10*num_columns, 5 * num_rows))

        for i, csv_file in enumerate(csv_files):
            model, _ = get_model_and_organs_info(csv_file, training_type='per_image_training')

            df = pd.read_csv(csv_file)
            values_gt = df[f'{metric}_gt']
            values_gt_e = df[f'{metric}_gt_e']
            values_gt_h = df[f'{metric}_gt_h']

            row_idx = i // num_columns
            col_idx = i % num_columns

            axs[row_idx, col_idx].plot(values_gt_e, label=f'{metric}_gt_e')
            axs[row_idx, col_idx].plot(values_gt_h, label=f'{metric}_gt_h')
            axs[row_idx, col_idx].plot(values_gt, label=f'{metric}_gt')
            axs[row_idx, col_idx].set_title(f'Approach: {model}')
            axs[row_idx, col_idx].set_xlabel('Iterations')
            axs[row_idx, col_idx].set_ylabel(f'{metric.upper()}')
            axs[row_idx, col_idx].legend()

        fig.suptitle(f'Organ: {organ} - Image: {id}')
        plt.savefig(os.path.join(outdir, f'{organ}_{id}_{metric.upper()}.png'))
        plt.close()

def generate_metrics_report(approach: str, 
                            indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', 
                            outdir='/home/modejota/Deep_Var_BCD/results_metrics_full_datasets/', 
                            organs=["Colon", "Breast", "Lung"], 
                            training_type='batch_training', 
                            metrics_to_use=["psnr", "mse", "ssim"],
                            use_inner_directory=True):
    
    indir += os.sep if indir[-1] != os.sep else ''
    outdir += os.sep if outdir[-1] != os.sep else ''
    if use_inner_directory:
        outdir_approach = f'{outdir}{approach}/batch_training' + os.sep if approach[-1] != os.sep else f'{outdir}{approach}/batch_training/'
    else:
        outdir_approach = outdir
    if not os.path.exists(outdir_approach):
        os.makedirs(outdir_approach)

    approach_path = f'{indir}{approach}' + os.sep if approach[-1] != os.sep else f'{indir}{approach}'
    csv_files = get_csv_filepaths(approach_path, training_type='batch_training')
    csv_files.sort()

    # print(f'Found {len(csv_files)} CSV files for {approach}.')
    print(f'Generating metrics report for {approach}.')
    if len(csv_files) == 0:
        return
    
    # Generate a map with the filenames for each organ
    organ_files = {organ: [] for organ in organs}
    for filename in csv_files:
        _, organ_info = get_model_and_organs_info(filename, training_type=training_type)
        organ_files[organ_info].append(filename)
          
    metrics_per_organ = {organ: {metric: [] for metric in metrics_to_use} for organ in organs}
    for organ in organs:
        for metric in metrics_to_use:
            for filename in organ_files[organ]:
                df = pd.read_csv(filename)
                max_value_gt = df[f'{metric}_gt'].max()
                idx_max_value_gt = df[f'{metric}_gt'].idxmax()
                value_gt_e = df[f'{metric}_gt_e'][idx_max_value_gt]
                value_gt_h = df[f'{metric}_gt_h'][idx_max_value_gt]

                metrics_per_organ[organ][metric].append((max_value_gt, value_gt_h, value_gt_e))
        
        # Calculate the mean value and the std for each metric
        for metric in metrics_to_use:
            metrics_per_organ[organ][metric] = np.array(metrics_per_organ[organ][metric])
            metrics_per_organ[organ][metric] = (metrics_per_organ[organ][metric].mean(axis=0), metrics_per_organ[organ][metric].std(axis=0))

    # Save the values into a text file
    with open(os.path.join(outdir_approach, f'{approach}_metrics.txt'), 'w') as f:
        # print("Saving metrics' report in", os.path.join(outdir_approach, f'{approach}_metrics.txt'))
        for organ in organs:
            f.write(f'Organ: {organ}\n')
            for metric in metrics_to_use:
                f.write(f'\t{metric.upper()}: {metrics_per_organ[organ][metric][0][0]} ± {metrics_per_organ[organ][metric][1][0]}\n')
                f.write(f'\t{metric.upper()}_GT_H: {metrics_per_organ[organ][metric][0][1]} ± {metrics_per_organ[organ][metric][1][1]}\n')
                f.write(f'\t{metric.upper()}_GT_E: {metrics_per_organ[organ][metric][0][2]} ± {metrics_per_organ[organ][metric][1][2]}\n')
            f.write('\n\n')

def generate_metric_report_all_methods(indir='/home/modejota/Deep_Var_BCD/results_full_datasets/',  outdir='/home/modejota/Deep_Var_BCD/results_metrics_full_datasets/'):
    directorios = []

    for nombre in os.listdir(indir):
        ruta_completa = os.path.join(indir, nombre)
        if os.path.isdir(ruta_completa) and not ruta_completa.endswith('graphs'):
            directorios.append(ruta_completa)

    for ruta_completa in directorios:
        generate_metrics_report(os.path.basename(ruta_completa), indir, outdir, use_inner_directory=False)

EXECUTE_SAMPLES = True
if __name__ == "__main__":

    if EXECUTE_SAMPLES:
        # print("Generating graphs for all approaches and selected images.")
        # generate_graphs_by_approach(indir='/home/modejota/Deep_Var_BCD/results_reduced_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_reduced_datasets/graphs/', training_type='per_image_training')
        
        print("Generating graphs for selected images.")
        generate_graphs_by_image(organ='Colon', id=0, indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_full_datasets/graphs/', training_type='batch_training')
        generate_graphs_by_image(organ='Colon', id=6, indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_full_datasets/graphs/', training_type='batch_training')
        generate_graphs_by_image(organ='Lung', id=0, indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_full_datasets/graphs/', training_type='batch_training')
        generate_graphs_by_image(organ='Lung', id=48, indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_full_datasets/graphs/', training_type='batch_training')
        generate_graphs_by_image(organ='Breast', id=0, indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_full_datasets/graphs/', training_type='batch_training')
        generate_graphs_by_image(organ='Breast', id=48, indir='/home/modejota/Deep_Var_BCD/results_full_datasets/', outdir='/home/modejota/Deep_Var_BCD/results_full_datasets/graphs/', training_type='batch_training')
        
        print("\nGenerating metrics' report for all methods.")
        generate_metric_report_all_methods()