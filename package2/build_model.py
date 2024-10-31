import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Đọc dữ liệu từ file CSV
ratings = pd.read_csv("data/ratings.csv", encoding='latin-1')
movies = pd.read_csv("data/movies.csv", encoding='latin-1')


# Hàm xử lý dữ liệu
def data_preprocessing():
    # Hợp nhất bảng 'ratings' và 'movies' theo cột 'movieId'
    rating_movies = pd.merge(ratings, movies, on="movieId")

    # Tạo bảng chéo (pivot table) với các hàng là 'title', các cột là 'userId' và giá trị là 'rating'
    # Điền giá trị 0 cho những ô trống (missing values)
    rating_movies_pivot = rating_movies.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    return rating_movies_pivot


# Hàm xem dữ liệu sau khi xử lý
def view_data_after_processing():
    # In ra dữ liệu đã được xử lý
    print(data_preprocessing())


# Hàm xuất dữ liệu đã xử lý ra file CSV
def explot_data_to_csv():
    # Lưu bảng chéo vào file 'new_data.csv'
    data_preprocessing().to_csv('data/new_data.csv')


# Hàm xây dựng mô hình hàng xóm gần nhất (Nearest Neighbors)
def build_model():
    # Tạo đối tượng NearestNeighbors với metric là 'cosine', thuật toán 'brute',
    # số lượng hàng xóm gần nhất là 7, và sử dụng tất cả các CPU hiện có (n_jobs=-1)
    model_nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=7, n_jobs=-1)

    # Huấn luyện mô hình với dữ liệu đã được xử lý
    model_nn.fit(data_preprocessing())

    return model_nn


# Hàm hiển thị các phim được đề xuất
def display_recommended_movie(input_name_movie, n_recommendations):
    # Lấy ra 'n' phim hàng xóm gần nhất với phim đầu vào dựa trên sự tương đồng
    indices = build_model().kneighbors(data_preprocessing().loc[[input_name_movie]], n_recommendations, return_distance = False)

    # In danh sách phim được đề xuất
    print("Recommended movies:")
    print("==================")

    # Chuyển chỉ số (indices) thành một mảng một chiều (1-dimensional array)
    flat_indices = indices.flatten()

    # Lặp qua các chỉ số và in tên phim tương ứng
    for index, value in enumerate(data_preprocessing().index[flat_indices]):
        print((index + 1), ". ", value)


# Hàm trả về danh sách các phim được đề xuất
def recommended_movie(input_name_movie, n_recommendations):
    # Lấy ra 'n' phim hàng xóm gần nhất với phim đầu vào
    indices = build_model().kneighbors(data_preprocessing().loc[[input_name_movie]], n_recommendations,
                                       return_distance=False)

    # Chuyển các chỉ số thành một mảng một chiều (1-dimensional array)
    flat_indices = indices.flatten()

    # Tạo danh sách các phim được đề xuất
    recommended_movies = [data_preprocessing().index[i] for i in flat_indices]

    return recommended_movies
