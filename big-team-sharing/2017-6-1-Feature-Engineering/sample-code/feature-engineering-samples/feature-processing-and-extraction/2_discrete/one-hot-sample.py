from sklearn.preprocessing import OneHotEncoder

source_data = [[0, 0, 3],
               [1, 1, 0],
               [2, 2, 1],
               [1, 0, 2]]

encoder = OneHotEncoder()
encoder.fit(source_data)

new_data = [[1, 2, 3]]
new_data_transformed = encoder.transform(new_data)

print(new_data_transformed.toarray())
