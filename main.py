from package1.view_data import *
from package2.build_model import *

print('Xem dữ liệu trước khi tiền xử lý')
View_data()
print('-'*70)

print('Xem dữ liệu sau tiền xử lý')
view_data_after_processing()
print('-'*70)

print('Xuất dữ liệu sau tiền xử lý thành file CSV')
explot_data_to_csv()
print('-'*70)



input_name_movie = str(input('Nhập tên movie:'))
n_recommendations = int(input('Nhập số phim muốn đề xuất:'))
display_recommended_movie(input_name_movie,n_recommendations)