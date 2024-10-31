# Module 2
import streamlit as st
import pandas as pd
from package2.build_model import recommended_movie

def main():
    df = pd.read_csv('data/new_data.csv')
    st.title('ỨNG DỤNG KHUYẾN NGHỊ PHIM')

    st.sidebar.header('Tùy chọn')
    keyword = st.sidebar.text_input('Nhập từ khóa')
    num_results = st.sidebar.number_input('Số lượng kết quả', min_value=1, max_value=len(df), value=5)

    if st.sidebar.button('Hiển thị tên phim'):
        results = df[df['title'].str.contains(keyword, case=False)]
        st.subheader(f'Tìm thấy {len(results)} kết quả cho "{keyword}"')
        st.table(results[['title']].head(num_results))

    if st.sidebar.button('Đề xuất phim'):
        recommended_movies = recommended_movie(keyword, num_results)
        st.subheader("Đề xuất phim:")
        for index, movie in enumerate(recommended_movies):
            st.write(f"{index+1}. {movie}")
    
if __name__ == '__main__':
    main()
