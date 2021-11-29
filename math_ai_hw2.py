from linear_regression_alternate_search import alternate_search
from my_io import build_testcase, read_dataset_with_pandas_to_nparray
from norm import find_error_x_from_y
from matplotlib import pyplot as plt

main_value = read_dataset_with_pandas_to_nparray('dataset/Admission_Predict.csv',6)
target_value = read_dataset_with_pandas_to_nparray('dataset/Admission_Predict.csv',-1)

number_of_value = len(main_value)
end_lenght_of_value, start_lenght_of_value = main_value.max(), main_value.min()
llenght_of_value = end_lenght_of_value - start_lenght_of_value
d_norm = 3
biasing = float(0)
weighting = float(0)

print(f'basic error is:\n{find_error_x_from_y(main_value,target_value,d_norm)}\n')
weighting, biasing = alternate_search(main_value, target_value, d_norm, printer = 0)
print(f'final a is {weighting} and c is {biasing} with error {find_error_x_from_y(main_value*weighting+biasing,target_value, d_norm)}')
weighting = round(weighting,9)
biasing = round(biasing,9)
print(f'final a is {weighting} and c is {biasing} with error {find_error_x_from_y(main_value*weighting+biasing,target_value, d_norm)}')

plt.plot(main_value,target_value,'.')
min_value = main_value.min()
max_value = main_value.max()
plt.plot([min_value,max_value], [min_value*weighting+biasing, max_value*weighting+biasing],'-')
plt.show()
print((main_value*weighting+biasing - target_value).max())