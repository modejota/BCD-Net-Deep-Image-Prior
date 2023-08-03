import os
import pandas as pd
import matplotlib.pyplot as plt

def get_csv_filepaths(directory_path='/home/modejota/Deep_Var_BCD/results/'):
    """Given a directory path, return a list of all the CSV file paths in that directory and its subdirectories
    Args:
        directory_path (str): The path to the directory
    Returns:
        list: A list of all the CSV file paths in that directory and its subdirectories
    """
    csv_filepaths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                csv_filepaths.append(os.path.join(root, file))
    
    return csv_filepaths

def get_certain_csv_filepaths(file_termination, directory_path='/home/modejota/Deep_Var_BCD/results/'):
    """Given a directory path, return a list of all the CSV file paths in that directory and its subdirectories
    Args:
        directory_path (str): The path to the directory
    Returns:
        list: A list of all the CSV file paths in that directory and its subdirectories
    """
    csv_filepaths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            filepath = os.path.abspath(os.path.join(root, file))
            if filepath.endswith(file_termination):
                csv_filepaths.append(os.path.join(root, file))
    
    return csv_filepaths

def get_model_and_organs_info(csv_filepath):
    """Given a CSV file path, extract the model name and organ info from the file path
    Args:
        csv_filepath (str): The path to the CSV file
    Returns:
        tuple: A tuple containing the model name and organ info
    """
    # Split the file path using the directory separator (\ in Windows)
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
                                metrics_to_use=["psnr", "mse", "ssim"]):
    """Generate one graph for each approach and metric given a directory containing the CSV files with the metrics for a set of images
    Args:
        indir (str): The path to the directory containing the CSV files
        outdir (str): The path to the directory where the graphs will be saved
        metrics_to_use (list): A list containing the metrics to use. psnr, mse and ssim are the only valid options
    """
    indir += os.sep if indir[-1] != os.sep else ''
    outdir += os.sep if outdir[-1] != os.sep else ''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    csv_files = get_csv_filepaths(indir)
    print(f'Found {len(csv_files)} CSV files.')
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
            model_name_, organ_info_ = get_model_and_organs_info(filepath)
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
            fig, ax = plt.subplots(num_rows, 3, figsize=(6*num_cols, 5 * num_rows))
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

def generate_graphs_by_image(organ: str, id: int, outdir='/home/modejota/Deep_Var_BCD/results/graphs/', metrics_to_use=["psnr", "mse", "ssim"]):
    """Generate one graph for each image and metric given a directory containing the CSV files with the metrics for a set of approaches
    Args:
        organ (str): The name of the organ
        id (str): The id of the image
        outdir (str): The path to the directory where the graphs will be saved
    """
    organ = organ.lower().capitalize()
    if id < 0:
        return
    filename = f'{organ}_{id}/metrics.csv'

    outdir += os.sep if outdir[-1] != os.sep else ''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    csv_files = get_certain_csv_filepaths(filename)
    csv_files.sort()

    num_files = len(csv_files)
    print(f'Found {num_files} CSV files for {organ}_{id}.')
    if num_files == 0:
        return

    for metric in metrics_to_use:
        num_rows = int(num_files**0.5)
        num_cols = (num_files + num_rows - 1) // num_rows

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5 * num_rows))

        for i, csv_file in enumerate(csv_files):
            model, _ = get_model_and_organs_info(csv_file)

            df = pd.read_csv(csv_file)
            values_gt = df[f'{metric}_gt']
            values_gt_e = df[f'{metric}_gt_e']
            values_gt_h = df[f'{metric}_gt_h']

            row_idx = i // num_cols
            col_idx = i % num_cols

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
