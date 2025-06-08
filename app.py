import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv('gojek_reviews_app5.csv')

# Title
st.title("Gojek App Review Sentiment Analysis")
st.markdown("Website ini mengklasifikasikan dan menganalisis ulasan pengguna Gojek berdasarkan sentimen: **positif**, **netral**, dan **negatif**. Hasil Kerja kelompok 10 UAS Penalaran Komputer")

# Sidebar
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Distribusi Sentimen", "Korelasi Rating & Sentimen", "WordCloud", "Panjang Ulasan", "Analisis Sentimen"])

# Page: Analisis Sentimen
if page == "Analisis Sentimen":
    st.subheader("Analisis Sentimen Ulasan Gojek")
    
    # Model selection
    model_option = st.selectbox("Pilih Model:", ["Traditional ML (SVM)", "Deep Learning (BiLSTM)"])
    
    # Text input for user review
    user_review = st.text_area("Masukkan ulasan:", "Aplikasi gojek sangat membantu dan mudah digunakan")
    
    if st.button("Analisis Sentimen"):
        # Placeholder for sentiment analysis logic
        # This is where you would implement the ML model prediction
        # For the sake of this example, we'll use dummy values.
        
        # Example sentiment analysis results (mock data)
        sentiment = "POSITIF"  # Example output
        confidence = {
            'negatif': 1.84,
            'netral': 5.20,
            'positif': 92.96
        }
        influential_words = ["mudah", "membantu", "aplikasi"]  # Example influential words
        
        # Display the results
        st.markdown(f"**Sentimen:** {sentiment}")
        st.markdown(f"**Confidence scores (estimated):**")
        st.bar_chart(confidence)
        
        st.markdown("**Influential Words:**")
        st.write(influential_words)
        
        st.markdown("**Highlighted Review:**")
        st.write(user_review)

# Page: Distribusi Sentimen
elif page == "Distribusi Sentimen":
    st.subheader("Distribusi Sentimen Ulasan Gojek") # Judul subheader lebih spesifik

    # Membuat figure dan axes matplotlib dengan ukuran yang lebih baik
    fig, ax = plt.subplots(figsize=(8, 6)) # Menyesuaikan ukuran plot

    # Menggunakan seaborn countplot untuk menampilkan distribusi sentimen
    # Pastikan 'df' di sini sudah berisi data dengan distribusi kustom Anda
    sns.countplot(
        data=df,
        x='stemmed',
        order=['negatif', 'netral', 'positif'], # Urutan kategori di sumbu X
        palette='viridis', # Pilihan palette warna, bisa diganti jika mau
        ax=ax
    )

    # Menambahkan judul dan label sumbu yang lebih deskriptif
    ax.set_title("Distribusi Sentimen Ulasan Gojek", fontsize=16)
    ax.set_xlabel("Kategori Sentimen", fontsize=12) # Label sumbu X
    ax.set_ylabel("Jumlah Ulasan", fontsize=12) # Label sumbu Y

    # Menambahkan jumlah di atas setiap batang (opsional, tapi sangat informatif)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', # Teks yang ditampilkan (jumlah)
                    (p.get_x() + p.get_width() / 2., p.get_height()), # Posisi teks
                    ha='center', va='center', # Penjajaran horizontal dan vertikal
                    xytext=(0, 10), # Offset dari puncak batang
                    textcoords='offset points',
                    fontsize=10) # Ukuran font teks

    # Menyesuaikan layout agar tidak ada label yang terpotong
    plt.tight_layout()

# Page: Korelasi Rating & Sentimen
elif page == "Korelasi Rating & Sentimen":
    st.subheader("Korelasi Rating dan Sentimen")

    # Pastikan kolom yang digunakan ada
    sentiment_col = 'label' if 'label' in df.columns else 'sentiment'

    # Hitung jumlah data berdasarkan score dan sentimen
    count_df = df.groupby(['score', sentiment_col]).size().reset_index(name='count')
    pivot_df = count_df.pivot(index='score', columns=sentiment_col, values='count').fillna(0)

    # Tampilkan dua kolom
    col1, col2 = st.columns([2, 1])

    with col1:
        # Grafik kurva
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_df.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Hubungan antara Rating (Score) dan Sentimen')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.legend(title='Sentimen')
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.markdown("**Jumlah Ulasan Tiap Kategori:**")
        st.dataframe(pivot_df.astype(int))

# Page: WordCloud
elif page == "WordCloud":
    st.subheader("WordCloud dan Grafik Frekuensi Kata")
    
    for senti in ['positif', 'netral', 'negatif']:
        st.markdown(f"### Sentimen **{senti.capitalize()}**")

        # Ambil teks berdasarkan sentimen
        text_series = df[df['label'] == senti]['review'].astype(str)
        all_words = " ".join(text_series).split()

        # Hitung frekuensi kata terbanyak (Top 10)
        word_freq = pd.Series(all_words).value_counts().head(10)

        # Buat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wc_image = wordcloud.generate_from_frequencies(word_freq.to_dict())

        # Buat layout dua kolom
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**WordCloud**")
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            ax_wc.imshow(wc_image, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        with col2:
            st.markdown("**Grafik Frekuensi Kata (Top 10)**")
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            ax_bar.barh(word_freq.index[::-1], word_freq.values[::-1], color='skyblue')
            ax_bar.set_xlabel("Frekuensi")
            ax_bar.set_title("10 Kata Paling Sering Muncul")
            st.pyplot(fig_bar)

# Page: Panjang Ulasan
elif page == "Panjang Ulasan":
    st.subheader("Distribusi Panjang Ulasan Berdasarkan Sentimen")
    df['review_length'] = df['review'].astype(str).apply(len)
    fig, ax = plt.subplots()
    for sentiment in ['positif', 'netral', 'negatif']:
        sns.kdeplot(df[df['label'] == sentiment]['review_length'], label=sentiment, ax=ax)
    ax.set_title("Distribusi Panjang Ulasan")
    ax.set_xlabel("Jumlah Karakter")
    ax.set_ylabel("Density")
    ax.legend(title="Sentimen")
    st.pyplot(fig)
