import pandas as pd
import numpy as np
import io
import os # Import the os module for path operations

# Define the file name based on the user's uploaded file
file_name = "World Economic Classifications v2.csv"

try:
    # Read the original CSV file into a pandas DataFrame
    # Assuming the file is accessible directly by its name in the environment
    original_df = pd.read_csv(file_name)
    print(f"Successfully loaded '{file_name}'.")
    print("\nOriginal DataFrame head:")
    print(original_df.head())
    print("\nOriginal DataFrame info:")
    original_df.info()

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    print("Please ensure the CSV file is uploaded and accessible.")
    # Exit or handle the error appropriately if the file is essential
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()

def generate_synthetic_data(df_template: pd.DataFrame, num_rows: int = 50) -> pd.DataFrame:
    """
    Generates synthetic data based on the column names and data types
    of a provided template DataFrame.

    Specific rules are applied for 'Country' and 'wealth_rank' columns.

    Args:
        df_template (pd.DataFrame): The DataFrame whose structure (column names, types)
                                    will be replicated.
        num_rows (int): The desired number of rows for the synthetic dataset.

    Returns:
        pd.DataFrame: A new DataFrame containing the generated synthetic data.
    """
    synthetic_df = pd.DataFrame() # Initialize an empty DataFrame for synthetic data

    # Iterate through each column of the original DataFrame to generate synthetic data
    for col_name in df_template.columns:
        original_series = df_template[col_name]

        # Special handling for the 'Country' column as requested by the user
        if col_name == 'country_name':
            synthetic_df[col_name] = [f'Country {i+1}' for i in range(num_rows)]
        # Special handling for the 'wealth_rank' column
        elif col_name == 'wealth_rank':
            # Ensure it's an integer only and within a reasonable range
            if not original_series.empty and pd.api.types.is_numeric_dtype(original_series):
                min_rank = int(original_series.min()) if pd.api.types.is_numeric_dtype(original_series) else 1
                max_rank = int(original_series.max()) if pd.api.types.is_numeric_dtype(original_series) else num_rows
                # Ensure min_rank is not greater than max_rank, and add 1 to max_rank for randint upper bound
                if min_rank > max_rank:
                    min_rank, max_rank = max_rank, min_rank # Swap if inverted
                # If min and max are the same, generate that specific value
                if min_rank == max_rank:
                    synthetic_df[col_name] = np.full(num_rows, min_rank, dtype=int)
                else:
                    synthetic_df[col_name] = np.random.randint(min_rank, max_rank + 1, num_rows)
            else:
                # Fallback if original 'wealth_rank' is empty or not numeric
                synthetic_df[col_name] = np.random.randint(1, num_rows * 2, num_rows) # Generate random integers up to num_rows * 2
            # Ensure the dtype is explicitly integer after generation
            synthetic_df[col_name] = synthetic_df[col_name].astype(int)

        # Handle numerical columns (integers and floats)
        elif pd.api.types.is_numeric_dtype(original_series):
            if not original_series.empty and original_series.min() is not None and original_series.max() is not None:
                min_val = original_series.min()
                max_val = original_series.max()

                if pd.api.types.is_integer_dtype(original_series):
                    # For integer types, generate random integers within the range
                    if min_val == max_val:
                        synthetic_df[col_name] = np.full(num_rows, min_val, dtype=int)
                    else:
                        synthetic_df[col_name] = np.random.randint(min_val, max_val + 1, num_rows)
                else: # For float types, generate random floats
                    synthetic_df[col_name] = np.random.uniform(min_val, max_val, num_rows)
            else:
                # Fallback for numerical columns that are empty or contain only NaNs
                synthetic_df[col_name] = np.random.rand(num_rows) * 100 # Default to random floats between 0-100
        # Handle categorical or object/string columns
        elif pd.api.types.is_object_dtype(original_series) or pd.api.types.is_string_dtype(original_series):
            unique_values = original_series.dropna().unique() # Get unique non-null values
            if unique_values.size > 0:
                # Sample randomly from the unique values of the original column
                synthetic_df[col_name] = np.random.choice(unique_values, num_rows)
            else:
                # Fallback for empty or all-NaN categorical columns
                synthetic_df[col_name] = [f'SyntheticCategory_{i % 5}' for i in range(num_rows)] # Generate generic categories
        else:
            # For any other unhandled data types, generate placeholder data
            print(f"Warning: Column '{col_name}' has an unhandled data type. Generating generic placeholder data.")
            synthetic_df[col_name] = ['Placeholder' for _ in range(num_rows)]

    return synthetic_df

# Generate 100 rows of synthetic data
num_synthetic_rows = 20
synthetic_data_df = generate_synthetic_data(original_df, num_rows=num_synthetic_rows)

print(f"\nGenerated {num_synthetic_rows} rows of Synthetic Data.")
print("\nSynthetic DataFrame head:")
print(synthetic_data_df.head())
print("\nSynthetic DataFrame info:")
synthetic_data_df.info()

# Save the synthetic data to a new CSV file in the Downloads folder
# This path is set for a macOS Downloads folder.
# It uses os.path.expanduser to correctly resolve the '~' (user's home directory).
output_file_path = os.path.expanduser("~/Downloads/synthetic_world_economic_classifications.csv")
synthetic_data_df.to_csv(output_file_path, index=False)
print(f"\nSynthetic data saved to: {output_file_path}")

# Note: When this code is executed in this environment, it runs within a sandboxed
# system that does not have access to your local machine's file system.
# The file will only be saved to your Downloads folder if you copy and run this
# Python script directly on your Mac.
