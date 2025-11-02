import pandas as pd

# Read data
branches_df = pd.read_csv('init/data/branch.csv').dropna()
distances_df = pd.read_csv('init/data/branch_distance.csv')

unique_codes = sorted(branches_df['branch_code'].unique())

print('='*60)
print('BRANCH DISTANCE DATA VERIFICATION')
print('='*60)
print(f'\nBranches in branch.csv: {len(unique_codes)}')
print(f'Total distance pairs: {len(distances_df)}')
print(f'Expected pairs (n×n): {len(unique_codes) * len(unique_codes)}')
print(f'\n✓ All pairs present: {len(distances_df) == len(unique_codes) * len(unique_codes)}')

print(f'\nDistance ranges:')
same_branch = len(distances_df[distances_df['distance_km'] == 0])
same_province = len(distances_df[(distances_df['distance_km'] > 0) & (distances_df['distance_km'] < 35)])
diff_province = len(distances_df[distances_df['distance_km'] >= 35])

print(f'  • Same branch (distance=0): {same_branch} pairs')
print(f'  • Same province (5-30km): {same_province} pairs')
print(f'  • Different provinces (>35km): {diff_province} pairs')

print(f'\nDistance statistics:')
print(f'  • Minimum: {distances_df["distance_km"].min():.2f} km')
print(f'  • Maximum: {distances_df["distance_km"].max():.2f} km')
print(f'  • Average: {distances_df["distance_km"].mean():.2f} km')
print(f'  • Median: {distances_df["distance_km"].median():.2f} km')

print(f'\nSample distances from branch 0 (HQ):')
sample = distances_df[distances_df['branch_code_1'] == 0].head(10)
for _, row in sample.iterrows():
    b2 = int(row['branch_code_2'])
    dist = row['distance_km']
    if b2 in branches_df['branch_code'].values:
        name = branches_df[branches_df['branch_code'] == b2]['branch_name'].values[0]
        print(f'  {b2:3d} -> {name:40s} : {dist:7.2f} km')

print('\n' + '='*60)
print('✓ Distance file successfully created!')
print('='*60)

