from linear_regression_alternate_search import alternate_search
from my_io import build_testcase, read_dataset_with_pandas_to_nparray, read_testcase
from norm import find_error_x_from_y
from matplotlib import pyplot as plt

# source_value = read_dataset_with_pandas_to_nparray('dataset/Admission_Predict.csv',6)
# target_value = read_dataset_with_pandas_to_nparray('dataset/Admission_Predict.csv',-1)
# while(1):
# build_testcase(1000,2,1000)
source_value, target_value, d_norm = read_testcase('dataset/mathAIH02.txt')

d_norm = 2
biasing = float(0)
weighting = float(0)

print(f'basic error is:\n{find_error_x_from_y(source_value,target_value,d_norm)}\n')
weighting, biasing = alternate_search(source_value, target_value, d_norm)
print(f'final a is {round(weighting,9)} and c is {round(biasing,9)} with error {find_error_x_from_y(source_value*weighting+biasing,target_value, d_norm)}')

# plt.plot(source_value,target_value,'.')
# min_value = source_value.min()
# max_value = source_value.max()
# plt.plot([min_value,max_value], [min_value*weighting+biasing, max_value*weighting+biasing],'-')
# plt.show()
if(find_error_x_from_y(source_value*weighting+biasing,target_value, d_norm) > 101):
    exit()