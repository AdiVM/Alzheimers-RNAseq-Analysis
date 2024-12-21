import pandas as pd
CellMatrixSyn16780177 = pd.read_parquet('/home/adm808/CellMatrixSyn16780177.parquet')

import numpy as np

# Assuming df is your gene expression DataFrame
# Rows are genes, columns are cells

# Step 1: Calculate CPM
cpm_df = CellMatrixSyn16780177.div(CellMatrixSyn16780177.sum(axis=0), axis=1) * 1e6

# Step 2: Apply log2 transformation
log2_cpm_df = np.log2(cpm_df + 1)

# Now log2_cpm_df contains the normalized and transformed data

log2_cpm_df.to_parquet('/home/adm808/NormalizedCellMatrixSyn16780177.parquet')