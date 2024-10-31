import pandas as pd
ratings = pd.read_csv("data/ratings.csv", encoding='latin-1')
movies = pd.read_csv("data/movies.csv", encoding='latin-1')

def View_data():
    print('--------Dữ liệu tập ratings------`----')
    print(ratings)
    print('Kích cỡ:')
    print(ratings.shape)
    print('Thông tin dữ liệu')
    print(ratings.info())
    print('Thống kê mô tả về tập Ratings:')
    print(ratings.describe())

    print('--------Dữ liệu tập movies----------')
    print(movies)
    print('Kích cỡ:')
    print(movies.shape)
    print('Thông tin dữ liệu')
    print(movies.info())
    print('Thống kê mô tả về tập Movies:')
    print(movies.describe())





