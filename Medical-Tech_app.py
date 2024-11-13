import streamlit as st
import base64
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(layout="wide")

def get_base64_image(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    return encoded_string

# Function to set up custom styling, including custom font size for st.write
def set_custom_styles(font_size="18px"):
    # Theme colors and font
    primary_color = "#11ccf5"         # Primary color
    background_color = "#d0d5d6"      # Main background color
    text_color = "#000000"            # Main text color (black)
    secondary_background = "#d0d5d6"  # Secondary background color
    font_family = "Serif"        # Font family

    # Custom CSS with dynamic font size
    st.markdown(f"""
    <style>
        /* Set main background color */
        .stApp {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font_family};
        }}

        /* Style for primary elements */
        .css-18e3th9 {{
            color: {primary_color};
        }}

        /* Style headers and specific content */
        h1, h2, h3, .stMarkdown, .stWrite {{
            color: {text_color};
            font-family: {font_family};
            font-size: {font_size};  /* Custom font size for st.write */
        }}

        /* Style secondary background areas */
        .main-container {{
            background-color: {secondary_background};
            border-radius: 10px;
            padding: 20px;
            margin: auto;
            color: {text_color};
        }}

        /* Style buttons, icons, and active elements */
        .css-1q8dd3e {{
            color: {primary_color};
        }}
    </style>
    """, unsafe_allow_html=True)

# Example usage
# Set custom font size for all `st.write` content
set_custom_styles(font_size="18px")


# Apply custom styles
set_custom_styles()

def main():
    logo_base64 = get_base64_image("logoo.png")
    st.markdown(
        f"""
        <style>
            .main-container {{
                max-width: 100%;
                margin-left: auto;
                margin-right: auto;
                padding: 10px; /* Reduced padding */
                background-color: #e6f7ff;
                border-radius: 10px;
            }}

            .center-content {{
                text-align: center;
                margin-top: -40px; /* Move content up */
            }}

            .center-content h1 {{
                margin-bottom: 10px;
                color: black;
            }}

            h2, h3, p, .stMarkdown, .stWrite {{
                color: black;
            }}

            /* Make elements with tabindex transparent */
            [tabindex] {{
                background-color: transparent !important;
            }}

            .custom-header {{
                background: rgba(0, 0, 0, 0); /* Transparent background */
                padding: 1px;
                border-radius: 1px;
                color: #000000; /* Change text color to match your design */
                text-align: center;
                border: 3px solid #11ccf5; /* Optional: add border to make it more visible */
                font-size: 23px; /* Set text size here */
                font-weight: bold; /* Make text bold */
                margin-bottom: 15px; /* Menambahkan jarak antara gambar dan teks header */
            }}

            .custom-heading {{
                font-size: 36px;
                color: #000000;
                font-family: 'Serif';
                text-align: center;
            }}
        </style>

        <div class="center-content">
            <!-- Display the logo above the text -->
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 400px; height: auto;">
            <h1 class="custom-heading">Website Deteksi Penyakit Pernafasan</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Horizontal menu
    tab_selection = option_menu(
        menu_title=None,
        options=["About Web", "Input & Predict Audio", "More Information", "Behind the Web"],
        orientation="horizontal",
        icons=["bi-check-square", "bi-search", "bi-book", "bi-bar-chart"],
        menu_icon="cast",
        default_index=0
    )

    if tab_selection == "About Web":
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: flex-start; border: 3px solid #11ccf5; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.5); max-width: 80%; margin: 10px 0;">
                <div style="flex: 1; padding-right: 20px;">
                    <img src="data:image/png;base64,{}" style="width: 100%; height: auto;">
                </div>
                <div style="flex: 2; text-align: justify; font-size: 1.5em; line-height: 1.6; color: #333;">
                    <p>
                        Selamat datang di Website Deteksi Penyakit Pernapasan, alat bantu berbasis AI yang dapat menganalisis suara napas Anda untuk mendeteksi kemungkinan penyakit pernapasan. Website deteksi penyakit pernapasan adalah platform digital yang membantu pengguna mendeteksi gejala-gejala penyakit pernapasan, seperti pneumonia, bronkitis, COPD dam URTI, secara cepat dan mudah. Dengan memanfaatkan teknologi pembelajaran mesin, website ini memungkinkan pengguna untuk mengunggah data suara pernafasan yang diambil menggunakan stetoskop yang kemudian akan dianalisis oleh sistem untuk mengidentifikasi kemungkinan adanya penyakit pernapasan.
                    </p>
                    <p>
                        Platform ini dirancang untuk memberikan hasil deteksi awal yang akurat, serta menyediakan informasi tentang langkah-langkah pencegahan dan perawatan yang dapat diambil. Selain itu, pengguna dapat berkonsultasi dengan tenaga medis melalui fitur chat atau telekonsultasi yang tersedia. Website ini mendukung upaya deteksi dini untuk meningkatkan kualitas kesehatan dan meminimalkan risiko komplikasi penyakit pernapasan.
                    </p>
                </div>
            </div>
            """.format(get_base64_of_bin_file("PARU-PARU1.png")),
            unsafe_allow_html=True
        )
    
            # Membuat frame untuk gambar yang diperbesar di sebelah kanan, dengan teks di bawah gambar
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: flex-end; border: 3px solid #11ccf5; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.5); max-width: 80%; margin: 20px 0; position: relative; float: right;">
                <div style="flex: 1; padding-left: 20px;">
                    <img src="data:image/png;base64,{}" style="width: 150%; height: auto;">
                </div>
                <div style="flex: 1; text-align: justify; font-size: 1.2em; line-height: 1.6; color: #333; margin-top: 15px;">
                    <p>
                        Gambar diagram diatas merupakan panduan singkat penggunaan website ini. Pengguna dapat masuk ke tab "Input & Predict Audio" untuk melakukan deteksi terhadap data. Pengguna dapat memasukkan rekaman suara pernafasan dengan menggunakan alat seperti stetoskop, setelah itu akan muncul tampilan spektogram mengenai audio tersebut. Setelah itu pengguna dapat klik tombol "Predict" sehingga sistem akan mengeluarkan hasil prediksi.
                    </p>
                    <p><strong>Disclaimer:</strong></p>
                    <ul style="text-align: justify; padding-left: 20px; font-size: 1.1em; line-height: 1.6; color: #333;">
                        <li>Pastikan lingkungan tenang, dan lakukan perekaman di area tanpa gangguan suara eksternal untuk merekam napas dengan benar demi hasil yang akurat.</li>
                        <li>Hasil prediksi dari website ini bukan pengganti diagnosis medis. Kami sarankan untuk berkonsultasi dengan tenaga medis profesional untuk pemeriksaan lebih lanjut.</li>
                    </ul>
                </div>
            </div>
            """.format(get_base64_of_bin_file("Penggunaan WEB.png")),
            unsafe_allow_html=True
        )
    
        # Membuat frame untuk teks pengantar dan tombol unduh audio dalam satu kerangka
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: flex-start; border: 3px solid #11ccf5; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.5); max-width: 80%; margin: 10px 0;">
                <div style="text-align: justify; font-size: 1.2em; line-height: 1.6; color: #333; margin-bottom: 15px;">
                    <p>
                        Dibawah ini merupakan beberapa sampel dataset audio pernafasan yang dapat diunduh untuk kemudian dicoba pada sistem prediksi jika kemungkinan Anda tidak memiliki data pernafasan. Pilih dan unduh data audio dalam format .wav berikut:

                </div>
            """,
            unsafe_allow_html=True
        )

        # Fungsi untuk membuat tombol unduh
        def download_audio_button(label, file_path, file_name):
            with open(file_path, "rb") as file:
                st.download_button(
                    label=label,
                    data=file,
                    file_name=file_name,
                    mime="audio/wav"
                )

        # Memanggil fungsi download_audio_button untuk setiap file dalam satu kolom dengan jarak antar tombol
        st.markdown('<div style="display: flex; flex-direction: column; gap: 10px;">', unsafe_allow_html=True)
        download_audio_button("Unduh Audio Penyakit Pernafasan Bronchiectasis", "Bronchiectasis.wav", "Bronchiectasis.wav")
        download_audio_button("Unduh Audio Penyakit Pernafasan COPD", "COPD.wav", "COPD.wav")
        download_audio_button("Unduh Audio Pernafasan yang Sehat", "HEALTHY.wav", "HEALTHY.wav")
        download_audio_button("Unduh Audio Penyakit Pernafasan Pneumonia", "Pneumonia.wav", "Pneumonia.wav")
        download_audio_button("Unduh Audio Penyakit Pernafasan URTI", "URTI.wav", "URTI.wav")
        st.markdown('</div>', unsafe_allow_html=True)

        # Menutup div utama untuk frame
        st.markdown("</div>", unsafe_allow_html=True)


    elif tab_selection == "Input & Predict Audio":
        gru_model = load_model('diagnosis_GRU_CNN_2.h5')
        classes = ["COPD (Chronic Obstructive Pulmonary Disease)", "Bronchiolitis", "Pneumonia", "URTI (Upper Respiratory Tract Infection)", "Healthy"]
        st.markdown('<div class="custom-header">Upload Audio</div>', unsafe_allow_html=True)
        st.write("Silakan Klik 'Browse files' untuk mengunggah file audio agar dapat mendiagnosis apakah terdapat penyakit pernapasan. Upload file audio dalam format WAV.")
        uploaded_file = st.file_uploader("", type="wav")

        if uploaded_file is not None:
            data_x, sampling_rate = librosa.load(uploaded_file, res_type='kaiser_fast')
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Waveform Audio")
                st.write("Berikut merupakan representasi visual dari sinyal audio yang menunjukkan bagaimana gelombang suara berfluktuasi seiring waktu.")
                fig_waveform, ax_waveform = plt.subplots(figsize=(6, 3))
                librosa.display.waveshow(data_x, sr=sampling_rate, ax=ax_waveform)
                ax_waveform.set(title="Waveform")
                st.pyplot(fig_waveform)

            with col2:
                st.subheader("Spektrogram Audio")
                st.write("Berikut merupakan kekuatan atau amplitudo suara di berbagai frekuensi yang berubah seiring waktu.")
                fig_spectrogram, ax_spectrogram = plt.subplots(figsize=(6, 3))
                S = librosa.feature.melspectrogram(y=data_x, sr=sampling_rate, n_mels=128, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, sr=sampling_rate, x_axis='time', y_axis='mel', ax=ax_spectrogram)
                fig_spectrogram.colorbar(img, ax=ax_spectrogram, format='%+2.0f dB')
                ax_spectrogram.set(title="Mel-Spektrogram")
                st.pyplot(fig_spectrogram)

            features = 52
            mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T, axis=0)
            val = np.expand_dims([mfccs], axis=1)
            
            # Display text above the button
            st.write("Klik tombol di bawah untuk memprediksi penyakit berdasarkan input audio")

            # Create the "Predict" button
            if st.button("Predict", key="predict"):
                # Generate prediction
                prediction = classes[np.argmax(gru_model.predict(val))]
                
                # Simpan hasil prediksi ke dalam session_state dengan kunci 'prediction'
                st.session_state['prediction'] = prediction

                st.markdown('<div class="custom-header">Hasil Prediksi</div>', unsafe_allow_html=True)

                # Create two columns
                col1, col2 = st.columns([1, 3])  # The first column will be smaller, the second larger

                with col1:
                    # Check if the prediction is "Healthy"
                    if prediction == "Healthy":
                        # Display the "Healthy" logo and text
                        st.image("logo_aman-removebg-preview.png", width=200, caption=" ", use_column_width=False)
                    else:
                        # Display caution image for other predictions
                        st.image("caution.png", width=200, caption=" ", use_column_width=False)

                with col2:
                    # Display the appropriate message above the prediction result
                    if prediction == "Healthy":
                        st.write("Dari audio nafas tersebut, anda diperkirakan yaitu")
                    else:
                        st.write("Dari audio nafas tersebut, anda diprediksi mengalami")
                    
                    # Display the prediction result in bold
                    st.markdown(f"#### **{prediction}**")  # Display the prediction in bold
                    
                    # Display the appropriate message below the prediction result
                    if prediction == "Healthy":
                        st.write("Pertahankan kesehatan pernafasan anda. Masuk ke tab 'More Information' untuk mendapatkan informasi seputar kesehatan pernafasan.")
                    else:
                        st.write("Masuk ke tab 'More Information' untuk mendapatkan informasi lebih lanjut mengenai penyakit ini")
        else:
            st.write(" ")
    
    elif tab_selection == "More Information":
        prediction = st.session_state.get('prediction')
        
        # Ambil prediksi dari session_state jika ada
        if prediction == "Healthy":
            st.markdown(f'<div class="custom-header">Tips-tips Menjaga Kesehatan Pernafasan</div>', unsafe_allow_html=True)
            # Define the URL and image
            Heal_img = "https://asset.kompas.com/crops/X9nxHMW_ugI5Suuf5QBC0XfNQCU=/0x0:900x600/230x153/data/photo/2023/08/10/64d4b5dfc11ee.png"
            Heal = "http://www.kompas.com/cekfakta/read/2023/08/11/111100182/infografik--kualitas-udara-jakarta-buruk-simak-tips-menjaga-kesehatan" 

            Heal_img2 = "https://asset.kompas.com/crops/S7aoLDGuUvgTiM3oK8TAzOAmgoA=/8x0:992x656/230x153/data/photo/2019/10/03/5d960dd28a311.jpg"
            Heal2 = "http://lifestyle.kompas.com/read/2020/06/17/133933420/9-cara-menjaga-organ-pernapasan-agar-tetap-sehat"

            Heal_img3 = "https://asset.kompas.com/crops/SQTnokNiKpvGvi65-EBz2Up0wzc=/0x0:461x307/230x153/data/photo/2021/07/29/61024c75c27bd.jpg"
            Heal3 = "http://lifestyle.kompas.com/read/2023/01/19/054046620/10-makanan-untuk-menjaga-fungsi-paru-paru-dan-kesehatan-pernapasan"

            Heal_img4 = "https://asset.kompas.com/crops/Z1my7BmlrLli9Cz0HoiPA2dM7Ts=/0x0:612x408/230x153/data/photo/2023/02/21/63f443d82b391.jpeg"
            Heal4 = "http://lifestyle.kompas.com/read/2023/06/08/135559920/6-tips-menjaga-kesehatan-anak-saat-kualitas-udara-jakarta-memburuk"


            # Create 2x2 layout with Streamlit columns
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            # First column content
            with col1:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Heal}" target="_blank" style="margin-right: 15px;">
                            <img src="{Heal_img}" alt="COPD Image" style="width: 600px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Heal}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">INFOGRAFIK: Kualitas Udara Jakarta Buruk, Simak Tips Menjaga Kesehatan Pernapasan</strong>
                            </a>
                            <p>Kualitas udara Jakarta beberapa waktu terakhir tercatat sebagai salah satu yang terburuk di dunia....</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Second column content
            with col2:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Heal2}" target="_blank" style="margin-right: 15px;">
                            <img src="{Heal_img2}" alt="COPD Image" style="width: 500px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Heal2}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">9 Cara Menjaga Organ Pernapasan agar Tetap Sehat</strong>
                            </a>
                            <p>Dengan melakukan berbagai cara menjaga organ pernapasan, maka kita sudah berperan dalam...</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Third column content
            with col3:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Heal3}" target="_blank" style="margin-right: 15px;">
                            <img src="{Heal_img3}" alt="COPD Image" style="width: 600px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Heal3}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">10 Makanan untuk Menjaga Fungsi Paru-paru dan Kesehatan Pernapasan</strong>
                            </a>
                            <p>Pastikan mengisi menu sehari-hari dengan makanan sehat agar paru-paru bekerja efektif, dan risiko....</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Fourth column content
            with col4:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Heal4}" target="_blank" style="margin-right: 15px;">
                            <img src="{Heal_img4}" alt="COPD Image" style="width: 500px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Heal4}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">6 Tips Menjaga Kesehatan Anak Saat Kualitas Udara Jakarta Memburuk</strong>
                            </a>
                            <p>Kualitas udara Jakarta memburuks sehingga membuat para orangtua mengkhawatirkan...</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )           
        
        elif prediction == "COPD (Chronic Obstructive Pulmonary Disease)":
            st.markdown(f'<div class="custom-header">Informasi Seputar Penyakit {prediction}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">1. Ringkasan</strong></p>
                        <p style="text-align: justify;">
                            COPD (Chronic Obstructive Pulmonary Disease) atau Penyakit Paru Obstruktif Kronik adalah penyakit paru-paru umum yang menyebabkan aliran udara terbatas dan masalah pernapasan. Penyakit ini terkadang disebut emfisema atau bronkitis kronis.
                        </p>
                        <p style="text-align: justify;">
                            Merokok dan polusi udara merupakan penyebab paling umum dari PPOK. Gejalanya dapat membaik jika seseorang menghindari rokok dan paparan polusi udara serta mendapatkan vaksin untuk mencegah infeksi.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">2. Gejala</strong></p>
                        <p style="text-align: justify;">
                            Gejala PPOK yang paling umum adalah kesulitan bernapas, batuk kronis (kadang-kadang disertai dahak), dan merasa lelah. Gejala PPOK dapat memburuk dengan cepat, yang sering kali memerlukan pengobatan tambahan.
                        </p>
                        <p style="text-align: justify;">
                            Orang dengan PPOK memiliki risiko lebih tinggi terhadap masalah kesehatan lainnya, termasuk:
                        </p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Infeksi paru-paru, seperti flu atau pneumonia</li>
                            <li>Kanker paru-paru</li>
                            <li>Masalah jantung</li>
                            <li>Otot lemah dan tulang rapuh</li>
                            <li>Depresi dan kecemasan</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">3. Penyebab</strong></p>
                        <p style="text-align: justify;">
                            PPOK berkembang secara bertahap dan disebabkan oleh kombinasi beberapa faktor risiko:
                        </p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Paparan tembakau</li>
                            <li>Paparan debu, asap, atau bahan kimia di tempat kerja</li>
                            <li>Polusi udara dalam ruangan</li>
                            <li>Peristiwa awal kehidupan seperti pertumbuhan yang buruk di dalam rahim</li>
                            <li>Kondisi genetik langka seperti defisiensi alfa-1 antitripsin</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">4. Perlakuan</strong></p>
                        <p style="text-align: justify;">
                            Perawatan PPOK termasuk:
                        </p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Pil steroid dan antibiotik</li>
                            <li>Oksigen untuk PPOK parah</li>
                            <li>Rehabilitasi paru</li>
                            <li>Operasi untuk memperbaiki gejala PPOK parah</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Menampilkan link sebagai teks yang dapat diklik
            st.markdown(
                """
                Sumber Bacaan 1: <a href="https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd)" target="_blank">Chronic obstructive pulmonary disease (COPD)</a>
                """, unsafe_allow_html=True
            )

            st.markdown(
                """
                Sumber Bacaan 2: <a href="https://lms.kemkes.go.id/courses/ce5a39bb-70c1-4713-aad8-5f0a7cd6e23c" target="_blank">Copd Care In Fpocus Advanced Chronic Obstructive Pulmonary Disease Rethinking</a>
                """, unsafe_allow_html=True
            )

            st.markdown(f'<div class="custom-header">Berita Terkait Penyakit {prediction}</div>', unsafe_allow_html=True)
            # Define the URL and image
            copd_image = "https://akcdn.detik.net.id/community/media/visual/2016/08/02/a602cf74-e13a-4600-800f-4102f19c4147_169.jpg?w=700&q=90"
            copd = "https://health.detik.com/berita-detikhealth/d-4799753/merokok-sejak-umur-belasan-waspadai-ppok-menyerang-di-bawah-umur-40"

            copd_image2 = "https://akcdn.detik.net.id/community/media/visual/2017/05/30/2706f40f-280d-4eb4-ba3c-16e86c95e2d9_43.jpg?w=300&q=80"
            copd2 = "https://health.detik.com/berita-detikhealth/d-7034522/15-november-2023-memperingati-hari-apa-ada-hari-ppok-sedunia"

            copd_image3 = "https://asset.kompas.com/crops/3S8J_qNDzvtIm2wk6I9leIUtcTo=/0x0:500x333/230x153/data/photo/2022/11/16/6374cb4f69885.jpg"
            copd3 = "http://health.kompas.com/read/22K16210100568/kenali-apa-itu-penyakit-paru-obstruktif-kronik-gejala-penyebabnya"

            copd_image4 = "https://akcdn.detik.net.id/community/media/visual/2018/04/18/4d3dd0d2-5bae-4aa8-91c4-af3bc9c39ac1_43.jpeg?w=300&q=80"
            copd4 = "https://health.detik.com/berita-detikhealth/d-3977515/barbara-bush-disebut-meninggal-akibat-ppok-apakah-itu"

            # Create 2x2 layout with Streamlit columns
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            # First column content
            with col1:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{copd}" target="_blank" style="margin-right: 15px;">
                            <img src="{copd_image}" alt="COPD Image" style="width: 900px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{copd}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Merokok Sejak Umur Belasan? Waspadai PPOK Menyerang di Bawah Umur 40</strong>
                            </a>
                            <p>Penyakit Paru Obstruktif Kronik (PPOK) adalah penyakit peradangan paru yang berkembang dalam jangka waktu panjang yang penyebab terbesarnya adalah merokok.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Second column content
            with col2:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{copd2}" target="_blank" style="margin-right: 15px;">
                            <img src="{copd_image2}" alt="COPD Image" style="width: 700px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{copd2}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">15 November 2023 Memperingati Hari Apa? Ada Hari PPOK Sedunia</strong>
                            </a>
                            <p>Salah satu momen kesehatan yang diperingati pada 15 November 2023 adalah Hari PPOK Sedunia. Seperti apa sejarah dan tema peringatannya tahun ini?</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Third column content
            with col3:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{copd3}" target="_blank" style="margin-right: 15px;">
                            <img src="{copd_image3}" alt="COPD Image" style="width: 500px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{copd3}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Kenali Apa itu Penyakit Paru Obstruktif Kronik, Gejala, Penyebabnya</strong>
                            </a>
                            <p>PPOK atau penyakit paru obstruktif kronik adalah salah satu gangguan pernapasan yang perlu.....</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Fourth column content
            with col4:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{copd4}" target="_blank" style="margin-right: 15px;">
                            <img src="{copd_image4}" alt="COPD Image" style="width: 700px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{copd4}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Barbara Bush Disebut Meninggal Akibat PPOK, Apakah Itu?</strong>
                            </a>
                            <p>Mantan Ibu Negara AS Barbara Bush baru saja dikabarkan meninggal dunia. Salah satu penyebabnya adalah penyakit paru obstruktif kronis, apakah itu?</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )           

        elif prediction == "Bronchiolitis":
            st.markdown(f'<div class="custom-header">Informasi Seputar Penyakit {prediction}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">1. Ringkasan</strong></p>
                        <p style="text-align: justify;">
                            Bronkiolitis adalah infeksi saluran napas yang menyebabkan radang dan penyumbatan di dalam bronkiolus. Kondisi ini merupakan penyebab umum sesak napas pada bayi dan anak usia 2 tahun ke bawah. Bronkiolus adalah saluran pernapasan terkecil di dalam paru-paru. Saat terjadi bronkiolitis, bronkiolus mengalami pembengkakan dan peradangan sehingga menyebabkan produksi lendir berlebih di saluran pernapasan.                        </p>
                        <p style="text-align: justify;">
                            Infeksi atau peradangan bisa disebabkan oleh beberapa jenis virus, termasuk di antaranya adalah virus flu dan pilek. Namun, jenis virus yang paling sering menyebabkan kondisi ini (terutama pada anak-anak yang masih berusia kurang dari dua tahun) adalah respiratory syncytial virus (RSV). Anak-anak biasanya tertular virus ketika berada di dekat dan terpapar oleh percikan liur, dari batuk atau bersin pengidap bronkiolitis.                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">2. Gejala</strong></p>
                        <p style="text-align: justify;">
                            Diagnosis penyakit bronkiolitis dilakukan dengan identifikasi dan mengobservasi anak melalui pemeriksaan fisik. Misalnya dengan mendengarkan suara paru-paru melalui stetoskop. Tidak hanya itu, dokter juga dapat menanyakan apakah anak memiliki tanda-tanda dehidrasi, seperti sering muntah dan menolak makan atau minum.                         </p>
                        <p style="text-align: justify;">
                            Gejala awal bronkiolitis adalah batuk, pilek atau hidung tersumbat, dan demam ringan. Beberapa hari kemudian, akan muncul keluhan lanjutan, seperti:                        </p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Batuk makin parah</li>
                            <li>Mengi</li>
                            <li>Sesak napas atau terlihat kesulitan untuk menarik napas</li>
                            <li>Sulit menyusu atau menelan</li>
                            <li>Muntah karena batuk</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">3. Penyebab</strong></p>
                        <p style="text-align: justify;">
                            Bronkiolitis umumnya disebabkan oleh respiratory syncytial virus (RSV). Virus ini biasanya menginfeksi anak usia 2 tahun ke bawah, terutama pada musim hujan. Selain RSV, influenza virus (virus penyebab flu) dan rhinovirus (virus penyebab batuk pilek) juga dapat menyebabkan bronkiolitis. Virus penyebab bronkiolitis sangat mudah menular. Anak dapat tertular virus ini jika tidak sengaja menghirup percikan liur dari orang yang bersin atau batuk karena flu atau batuk pilek. Penularan juga bisa terjadi jika anak menyentuh mulut atau hidung dengan tangan yang terkontaminasi virus dari barang-barang di sekitarnya                        <p style="text-align: justify;">
                        <p style="text-align: justify;">
                            Berikut ini adalah beberapa kondisi yang dapat meningkatkan risiko seorang anak terkena bronkiolitis:<ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Memiliki daya tahan tubuh yang lemah</li>
                            <li>Terlahir prematur</li>
                            <li>Berusia kurang dari 3 bulan</li>
                            <li>Tidak pernah atau kurang mendapatkan ASI</li>
                            <li>Tinggal di lingkungan padat penduduk</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">4. Perlakuan</strong></p>
                        <p style="text-align: justify;">
                            Pengobatan bronkiolitis bertujuan untuk meredakan gejala yang dialami anak, seperti kesulitan bernapas atau mengi. Jika anak menderita bronkiolitis yang tidak tergolong parah, biasanya dokter akan menyarankan perawatan di rumah, seperti:                        </p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Memastikan anak beristirahat dengan cukup</li>
                            <li>Memberikan cukup ASI atau susu formula jika anak masih berusia 1 tahun ke bawah</li>
                            <li>Memberikan asupan cairan yang cukup pada anak, bisa dengan minum air putih atau cairan elektrolit</li>
                            <li>Menjaga kelembapan udara kamar anak, misalnya dengan memasang humidifier</li>
                            <li>Menjauhkan anak dari polusi udara, terutama asap rokok</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Menampilkan link sebagai teks yang dapat diklik
            st.markdown(
                """
                Sumber Bacaan 1: <a href="https://www.alodokter.com/bronkiolitis" target="_blank">Bronkiolitis</a>
                """, unsafe_allow_html=True
            )

            st.markdown(
                """
                Sumber Bacaan 2: <a href="https://www.halodoc.com/kesehatan/bronkiolitis?srsltid=AfmBOooYlt0xre1wU4bmkOIx0XFoxpKoNkLxEmos9BCLaad8nm6DlAh4" target="_blank">Bronkiolitis</a>
                """, unsafe_allow_html=True
            )

            st.markdown(f'<div class="custom-header">Berita Terkait Penyakit {prediction}</div>', unsafe_allow_html=True)
            # Define the URL and image
            Image_bro = "https://asset.kompas.com/crops/WOtgoQXmXyDYdRIZrud1V2tFceY=/0x0:612x408/490x326/data/photo/2023/07/26/64c0d6191ada8.jpeg"
            Bro = "https://www.kompas.com/tren/read/2023/07/26/193000565/10-gejala-bronkiolitis-pada-anak-yang-perlu-diwaspadai-apa-saja-"

            Image_bro2 = "https://asset.kompas.com/crops/-oedEzXWRetlsuy9arBXUR2ZcVE=/0x0:1000x667/230x153/data/photo/2021/11/04/61836073ef58c.jpg"
            Bro2 = "https://health.kompas.com/read/2021/11/05/150000668/menyerang-paru-paru-apa-perbedaan-bronkiolitis-dan-bronkitis-"

            Image_bro3 = "https://asset.kompas.com/crops/O7d5B95W6nEKXPHoKoYrcka6z4o=/134x0:719x390/1200x800/data/photo/2016/05/12/1159144ThinkstockPhotos-462119163780x390.jpg"
            Bro3 = "https://health.kompas.com/read/2021/10/28/110300568/4-penyebab-bronkiolitis-yang-perlu-diwaspadai"

            Image_bro4 = "https://d1vbn70lmn1nqe.cloudfront.net/prod/wp-content/uploads/2024/11/04021932/Bronkiolitis-Infeksi-Paru-Paru-pada-Bayi-yang-Punya-Gejala-Seperti-Flu.jpg.webp"
            Bro4 = "https://www.halodoc.com/artikel/bronkiolitis-infeksi-paru-paru-pada-bayi-yang-punya-gejala-seperti-flu?srsltid=AfmBOoohUffr3WSQFy0L-vvtg7U-jvHzSDLYrSpcHYGorzJKu8cJDpMC" 

            # Create 2x2 layout with Streamlit columns
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            # First column content
            with col1:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Bro}" target="_blank" style="margin-right: 15px;">
                            <img src="{Image_bro}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Bro}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">10 Gejala Bronkiolitis pada Anak yang Perlu Diwaspadai, Apa Saja?</strong>
                            </a>
                            <p>Bronkiolitis adalah infeksi paru-paru yang umumnya menyerang anak kecil dan bayi. Kenali sejumlah gejalanya berikut ini.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Second column content
            with col2:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Bro2}" target="_blank" style="margin-right: 15px;">
                            <img src="{Image_bro2}" alt="COPD Image" style="width: 500px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Bro2}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Menyerang Paru-paru, Apa Perbedaan Bronkiolitis dan Bronkitis?</strong>
                            </a>
                            <p>Sama-sama menyerang paru-paru, apa perbedaan bronkiolitis dan bronkitis?</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Third column content
            with col3:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Bro3}" target="_blank" style="margin-right: 15px;">
                            <img src="{Image_bro3}" alt="COPD Image" style="width: 700px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Bro3}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">4 Penyebab Bronkiolitis yang Perlu Diwaspadai</strong>
                            </a>
                            <p>Penyebab bronkiolitis, infeksi saluran udara terkecil di paru-paru (bronkiolus) pada anak-anak dan...</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Fourth column content
            with col4:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Bro4}" target="_blank" style="margin-right: 15px;">
                            <img src="{Image_bro4}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Bro4}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Bronkiolitis: Infeksi Paru-Paru pada Bayi yang Punya Gejala Seperti Flu</strong>
                            </a>
                            <p>Bronkiolitis pada bayi bisa menyebabkan kekurangan kadar oksigen dalam darah hingga gagal napas yang bisa berakibat fatal.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )           

        elif prediction == "Pneumonia":
            st.markdown(f'<div class="custom-header">Informasi Seputar Penyakit {prediction}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">1. Ringkasan</strong></p>
                        <p style="text-align: justify;">
                            Pneumonia adalah peradangan paru-paru yang disebabkan oleh infeksi. Pneumonia bisa menimbulkan gejala yang ringan hingga berat. Beberapa gejala yang umum dialami penderita pneumonia adalah batuk berdahak, demam, dan sesak napas.                        </p>
                        <p style="text-align: justify;">
                            Pneumonia juga dikenal dengan istilah paru-paru basah. Pada kondisi ini, infeksi menyebabkan peradangan pada kantong-kantong udara (alveoli) di salah satu atau kedua paru-paru. Akibatnya, alveoli dipenuhi cairan atau nanah sehingga membuat penderitanya sulit bernapas. Badan Kesehatan Dunia (WHO) memperkirakan sebanyak 15 persen kematian anak-anak usia balita di seluruh dunia terkait dengan pneumonia. Meskipun begitu, pneumonia bisa menimpa orang dewasa dengan dampak yang kurang lebih sama.                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">2. Gejala</strong></p>
                        <p style="text-align: justify;">
                            Pada dasarnya, gejala pneumonia hampir sama dengan masalah paru-paru lainnya, di antaranya batuk dengan intensitas tinggi dan disertai dahak. Selain itu, dilansir dalam Mayo Clinic, berikut beberapa gejala umum yang terjadi saat Anda mengalami pneumonia:                        </p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Demam tinggi, suhu tubuh mencapai lebih dari 38 derajat Celcius</li>
                            <li>Dada terasa sakit dan sulit bernapas</li>
                            <li>Penurunan nafsu makan</li>
                            <li>Berkeringat</li>
                            <li>Mengigil</li>
                            <li>Detak jantung terasa cepat</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">3. Penyebab</strong></p>
                        <p style="text-align: justify;">
                            Penyebab dari pneumonia beragam, tetapi berdasarkan organisme dan tempat penyebarannya, pneumonia dibedakan menjadi dua. Pertama ada pneumonia komunitas yang penyebarannya terjadi di komunitas (lingkungan umum) dan pneumonia yang ditularkan di rumah sakit.<p style="text-align: justify;">
                            Contoh organisme yang menyebabkan pneumonia yang ditularkan di tempat umum, antara lain:<ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Bakteri, yang paling sering adalah Streptococcus pneumoniae.</li>
                            <li>Organisme yang menyerupai bakteri, Mycoplasma pneumonia.</li>
                            <li>Jamur, biasanya jamur akan menyerang orang dengan gangguan sistem imun.</li>
                            <li>Virus</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">4. Perlakuan</strong></p>
                        <p style="text-align: justify;">
                            Pengobatan pneumonia akan disesuaikan dengan penyebab dan tingkat keparahan yang dialami pasien. Pneumonia akibat infeksi bakteri akan ditangani dengan obat antibiotik. Dokter juga dapat memberikan obat pneumonia lain untuk meredakan gejala batuk, demam, atau nyeri.
                        <p style="text-align: justify;">
                            Pneumonia dapat dicegah dengan beberapa cara, di antaranya menjalani vaksinasi, menjaga kebersihan diri, misalnya rajin mencuci tangan dan tidak menyentuh hidung atau mulut dengan tangan yang belum dicuci, dan menghindari kontak dengan orang yang sedang sakit</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Menampilkan link sebagai teks yang dapat diklik
            st.markdown(
                """
                Sumber Bacaan 1: <a href="https://www.siloamhospitals.com/informasi-siloam/artikel/pneumonia" target="_blank">Pneumonia (Radang Paru-Paru) - Gejala dan Cara Mengobatinya</a>
                """, unsafe_allow_html=True
            )

            st.markdown(
                """
                Sumber Bacaan 2: <a href="https://www.alodokter.com/pneumonia" target="_blank">Pneumonia - Gejala, Penyebab, dan Pengobatan</a>
                """, unsafe_allow_html=True
            )
            
            st.markdown(
                """
                Sumber Bacaan 3: <a href="https://www.halodoc.com/kesehatan/pneumonia?srsltid=AfmBOoorQqiGUTr5opLnIGbioEPtK0eSxNIjrqORXPUcDRjW0h-55yKs" target="_blank">Pneumonia - Gejala, Penyebab, Pencegahan & Pengobatan</a>
                """, unsafe_allow_html=True
            )

            st.markdown(f'<div class="custom-header">Berita Terkait Penyakit {prediction}</div>', unsafe_allow_html=True)
            # Define the URL and image
            Pneu_img = "https://akcdn.detik.net.id/community/media/visual/2021/09/02/pacar-cristiano-ronaldo-georgina-rordiguez_43.jpeg?w=300&q=80"
            Pneu =  "https://health.detik.com/berita-detikhealth/d-7615474/pacar-cristiano-ronaldo-dirawat-di-rs-karena-pneumonia-begini-kondisinya"

            Pneu_img2 = "https://akcdn.detik.net.id/community/media/visual/2023/02/22/ilustrasi-pneumonia-yang-disebabkan-oleh-bakteri_43.jpeg?w=300&q=80"
            Pneu2 = "https://www.detik.com/jabar/berita/d-7574169/mengenal-dampak-fatal-dari-paru-paru-basah-yang-kamu-perlu-tahu"

            Pneu_img3 = "https://asset.kompas.com/crops/JE-Gxln_5GhIjuNhevX8dJoc0K4=/0x98:696x562/230x153/data/photo/2020/03/31/5e82c476767ab.png"
            Pneu3 = "https://health.kompas.com/read/24I14204000868/dokter-luruskan-mitos-mandi-malam-hari-sebabkan-pneumonia"

            Pneu_img4 = "https://asset.kompas.com/crops/7K9dsF1xQBnewQRC1zMV0VJlykc=/0x0:780x520/490x326/data/photo/2023/11/27/65647a7e7a76a.jpg"
            Pneu4 = "https://lifestyle.kompas.com/read/2024/11/08/174243420/hati-hati-riwayat-pneumonia-pada-anak-bisa-ganggu-tumbuh-kembangnya"


            # Create 2x2 layout with Streamlit columns
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            # First column content
            with col1:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Pneu}" target="_blank" style="margin-right: 15px;">
                            <img src="{Pneu_img}" alt="COPD Image" style="width: 600px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Pneu}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Pacar Cristiano Ronaldo Dirawat di RS karena Pneumonia, Begini Kondisinya</strong>
                            </a>
                            <p>Georgina Rodriguez, kekasih Cristiano Ronaldo, dirawat empat hari karena pneumonia. Dia membagikan kondisi terkini.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Second column content
            with col2:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Pneu2}" target="_blank" style="margin-right: 15px;">
                            <img src="{Pneu_img2}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Pneu2}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Mengenal Dampak Fatal dari Paru-paru Basah yang Kamu Perlu Tahu</strong>
                            </a>
                            <p>Spesialis penyakit paru, dr. Naindra Kemala Dewi, Sp.P menjelaskan dampak dari terjadinya penyakit paru-paru basah atau pneumonia. Simak video penjelasannya.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Third column content
            with col3:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Pneu3}" target="_blank" style="margin-right: 15px;">
                            <img src="{Pneu_img3}" alt="COPD Image" style="width: 500px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Pneu3}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Dokter Luruskan Mitos Mandi Malam Hari Sebabkan Pneumonia</strong>
                            </a>
                            <p>Penyebab pneumonia dan juga paru-paru basah bukanlah karena mandi malam atau kena semprot</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Fourth column content
            with col4:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{Pneu4}" target="_blank" style="margin-right: 15px;">
                            <img src="{Pneu_img4}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{Pneu4}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Hati-hati, Riwayat Pneumonia pada Anak Bisa Ganggu Tumbuh Kembangnya</strong>
                            </a>
                            <p>Anak dengan riwayat pneumonia, tumbuh kembangnya bisa terganggu. Ini karena, pneumonia adalah infeksi yang mengganggu penyerapan gizi.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                ) 

        elif prediction == "URTI (Upper Respiratory Tract Infection)":
            st.markdown(f'<div class="custom-header">Informasi Seputar Penyakit {prediction}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">1. Ringkasan</strong></p>
                        <p style="text-align: justify;">
                            Infeksi saluran pernapasan adalah infeksi yang terjadi di saluran pernapasan, bisa atas atau bawah. Meski biasanya dapat sembuh tanpa perawatan khusus, kondisi ini bisa berbahaya bagi anak-anak, lansia, atau orang dengan daya tahan tubuh yang lemah. Infeksi saluran napas bawah terjadi di bronkus, bronkiolus, atau paru-paru. Infeksi ini dapat disebabkan oleh banyak hal, mulai dari bakteri, virus, dan dapat menimbulkan sejumlah gejala sepertidemam, batuk pilek, sakit tenggorokan, gangguan penciuman, sakit kepala, pegal-pegal, sampai sesak napas.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">2. Gejala</strong></p>
                        <p style="text-align: justify;">
                            Meskipun biasanya bersifat ringan dan bisa pulih dengan sendirinya, namun ada juga kasus yang berujung berat. Oleh karena itu, penyakit ini tidak bisa terlalu dianggap enteng. Berikut ini adalah beberapa gejala yang kerap dialami saat terkena infeksi saluran pernapasan atas:</p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Batuk</li>
                            <li>Bersin</li>
                            <li>Hidung berlendir</li>
                            <li>Hidung tersumbat</li>
                            <li>Demam</li>
                            <li>Tenggorokan gatal atau sakit</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            

            with col1:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">3. Penyebab</strong></p>
                        <p style="text-align: justify;">
                            Infeksi saluran pernapasan adalah infeksi yang terjadi di saluran pernapasan, bisa atas atau bawah. Meski biasanya dapat sembuh tanpa perawatan khusus, kondisi ini bisa berbahaya bagi anak-anak, lansia, atau orang dengan daya tahan tubuh yang lemah. Infeksi saluran napas bawah terjadi di bronkus, bronkiolus, atau paru-paru. Infeksi ini dapat disebabkan oleh banyak hal, mulai dari bakteri, virus, dan dapat menimbulkan sejumlah gejala sepertidemam, batuk pilek, sakit tenggorokan, gangguan penciuman, sakit kepala, pegal-pegal, sampai sesak napas.</p>
                        <p style="text-align: justify;">
                            Virus atau bakteri penyebab infeksi saluran pernapasan juga bisa masuk ke dalam tubuh akibat tidak sengaja menyentuh mata, hidung, atau mulut dengan tangan yang belum bersih, terlebih setelah menyentuh permukaan benda yang sudah terpapar virus atau bakteri. Berikut ini adalah beberapa jenis kuman yang paling sering menyebabkan infeksi saluran pernapasan:</p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Virus seperti Rhinovirus, Virus herpes simplex, Hantavirus dan Paramyxovirus</li>
                            <li>Bakteri misalnya Streptococcus grup A, Corynebacteroum diphteriae, Neiseria gonorrhoeae dan Mycoplasma pneumoniae</li>
                            <li>Parasit, seperti Pneumocytis carinii</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                        <p><strong style="font-size: 1.4em;">4. Perlakuan</strong></p>
                        <p style="text-align: justify;">
                            Pengobatan infeksi saluran pernapasan akan disesuaikan dengan kondisi pasien. Sebagian kasus infeksi saluran pernapasan yang disebabkan oleh virus terkadang tidak membutuhkan perawatan spesifik dan bisa sembuh dengan sendirinya. Namun, untuk membantu meredakan gejalanya, pasien dapat melakukan beberapa penanganan mandiri berikut:</p>
                        <ul style="text-align: justify; padding-left: 20px; font-size: 1em;">
                            <li>Beristirahat yang cukup</li>
                            <li>andi atau berendam dengan air hangat</li>
                            <li>Mengonsumsi makanan atau minuman yang hangat</li>
                            <li>Minum air putih dalam jumlah yang cukup</li>
                            <li>Menghindari paparan udara dingin</li>
                            <li>Berkumur dengan air garam</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Menampilkan link sebagai teks yang dapat diklik
            st.markdown(
                """
                Sumber Bacaan 1: <a href="https://www.alodokter.com/infeksi-saluran-pernapasan" target="_blank">Infeksi Saluran Pernapasan</a>
                """, unsafe_allow_html=True
            )

            st.markdown(
                """
                Sumber Bacaan 2: <a href="https://www.jagadiri.co.id/news/infeksi-saluran-pernapasan-atas" target="_blank">Infeksi Saluran Pernapasan Atas Menular atau Tidak?</a>
                """, unsafe_allow_html=True
            )

            st.markdown(f'<div class="custom-header">Berita Terkait Penyakit {prediction}</div>', unsafe_allow_html=True)
            # Define the URL and image

            urti_image = "https://asset.kompas.com/crops/h33TvVa7QBHjQZWwlnAT5eGp3Ik=/0x1:500x334/490x326/data/photo/2020/03/26/5e7c62e270f44.jpg"
            urti = "https://megapolitan.kompas.com/read/2021/12/21/16170441/1793-warga-kota-bekasi-derita-urti-pada-2021-jumlah-kasus-diklaim"

            urti_image2 = "https://akcdn.detik.net.id/community/media/visual/2024/09/27/wakil-ketua-pansus-pertambangan-dpr-aceh-irpannusir-geram-dengan-kondisi-udara-meulaboh-tercemar-akibat-pertambangan-foto-isti_43.jpeg?w=300&q=80"
            urti2 = "https://www.detik.com/sumut/berita/d-7561533/dpra-geram-udara-meulaboh-tercemar-akibat-tambang-debu-kimia-di-mana-mana"

            urti_image3 = "https://akcdn.detik.net.id/community/media/visual/2023/04/23/potret-thailand-dilanda-panas-ekstrem-suhu-bisa-capai-45c-2_43.jpeg?w=300&q=80"
            urti3 = "https://health.detik.com/berita-detikhealth/d-7377676/cuaca-panas-bikin-kasus-ispa-naik-begini-kata-dokter-paru"

            urti_image4 = "https://akcdn.detik.net.id/community/media/visual/2024/05/04/ribuan-warga-terdampak-erupsi-gunung-ruang-dievakuasi_43.jpeg?w=300&q=80"
            urti4 = "https://news.detik.com/berita/d-7327164/ispa-hingga-dermatitis-jangkiti-warga-terdampak-erupsi-gunung-ruang"

            # Create 2x2 layout with Streamlit columns
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            # First column content
            with col1:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{urti}" target="_blank" style="margin-right: 15px;">
                            <img src="{urti_image}" alt="COPD Image" style="width: 900px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{urti}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">1.793 Warga Kota Bekasi Derita URTI pada 2021, Jumlah Kasus Diklaim Berkurang</strong>
                            </a>
                            <p>Kepala Dinkes Kota Bekasi Tanti Rohilawati mengatakan, meskipun terbilang cukup banyak, namun terjadi penurunan kasus jika dibanding 2019 dan 2020.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Second column content
            with col2:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{urti2}" target="_blank" style="margin-right: 15px;">
                            <img src="{urti_image2}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{urti2}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">DPRA Geram Udara Meulaboh Tercemar Akibat Tambang: Debu Kimia di Mana-mana</strong>
                            </a>
                            <p>Anggota DPRA Irpannusir geram dengan kondisi udara Meulaboh tercemar akibat pertambangan. Dia sebut daerah itu saat ini lebih cocok diberi gelar 'petro kimia'.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Third column content
            with col3:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{urti3}" target="_blank" style="margin-right: 15px;">
                            <img src="{urti_image3}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{urti3}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">Cuaca Panas Bikin Kasus ISPA Naik? Begini Kata Dokter Paru</strong>
                            </a>
                            <p>Dokter paru mengungkapkan cuaca panas beberapa waktu terakhir berisiko memicu sejumlah penyakit, salah satunya ISPA. Begin gejala yang perlu diwaspadai.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Fourth column content
            with col4:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <a href="{urti4}" target="_blank" style="margin-right: 15px;">
                            <img src="{urti_image4}" alt="COPD Image" style="width: 800px; border-radius: 5px;">
                        </a>
                        <div style="text-align: justify;">
                            <a href="{urti4}" target="_blank" style="text-decoration: none; color: inherit;">
                                <strong style="font-size: 1.1em;">ISPA hingga Dermatitis Jangkiti Warga Terdampak Erupsi Gunung Ruang</strong>
                            </a>
                            <p>Gunung Ruang di Sulawesi Utara, mengalami erupsi beberapa waktu terakhir. Kementerian Kesehatan melaporkan ribuan orang yang terdampak erupsi terluka.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )  

        else:
            st.write("Upload dan predict terlebih dahulu audio yang anda inginkan untuk mendapatkan informasi lebih lanjut")

    elif tab_selection == "Behind the Web":
        st.markdown(f'<div class="custom-header">Sekilas Tentang Developer Web</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                    <p><strong style="font-size: 1.4em;">Sekilas Tentang Saya</strong></p>
                    <p style="text-align: justify;">
                        Saya Miko Ardian, seorang Junior Data Scientist dengan minat yang kuat dalam analisis dan pemodelan data untuk menghasilkan wawasan dan keputusan bisnis. Saya memiliki pengalaman dalam beberapa proyek seperti analisis big data, komunikasi data melalui dasbor, pembelajaran mesin, dan statistik. Saya selalu mengambil inisiatif dalam kelompok, menunjukkan disiplin, dan melaksanakan tugas dengan pendekatan yang terstruktur. Selain itu, saya memiliki keterampilan komunikasi yang sangat baik dan senang berkembang dalam lingkungan kerja tim yang kolaboratif.                   </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                    <p><strong style="font-size: 1.4em;">Riwayat Pendidikan dan Tempat Bekerja</strong></p>
                    <p style="text-align: justify;">
                        Berkuliah di Telkom Unversity Kampus Purwokerto dengan mengambil jurusan S1 Data Science, adapun IPK yaitu 3,95 dengan berbeagai mata kuliah meliputi Machine Learning, Deep Learning, Natural Language Processing, Big Data Analytics, Data Science & Artificial Intelligence, Statistical Methods, Data Visualization, Data Science Project</p>
                    <p style="text-align: justify;">
                        Bekerja di PT.Penerbit Erlangga sebagai Junior Data Science pada pertengahan Juni sampai dengan saat ini. Adapun jobdesc yang dilakukan antara lain mengembangkan model machine learning, menganalisis data dan menampilkannya pada stakeholder, mengembangkan project AI dan mengatur API model machine learning kedalam website dan aplikasi perusahaan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(f'<div class="custom-header">Sekilas Tentang Algoritma yang Dipakai</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                    <p><strong style="font-size: 1.4em;">Sekilas Tentang Project</strong></p>
                    <p style="text-align: justify;">
                        Klasifikasi penyakit pernapasan dari data suara dapat dilakukan dengan menggabungkan arsitektur Convolutional Neural Network (CNN) dan Gated Recurrent Unit (GRU). CNN dapat mengekstrak fitur spasial dari spektrum suara, sementara GRU dapat menangkap dependensi jangka panjang dari data sekuensial, seperti aliran waktu dalam suara.                   </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                    <p><strong style="font-size: 1.4em;">Sekilas Tentang Algoritma</strong></p>
                    <p style="text-align: justify;">
                        Seperti yang telah di jelaskan bahwa project ini menggunakan kombinasi algoritma CNN dan GRU.                         Kombinasi CNN dan GRU banyak digunakan dalam tugas-tugas di mana diperlukan pemrosesan data spasial dan temporal sekaligus, seperti dalam analisis sinyal suara atau video. CNN digunakan untuk mengekstraksi fitur spasial, seperti pola atau tekstur pada data (contoh: spektrogram dari suara) sedangkan GRU digunakan untuk memahami urutan temporal dari fitur-fitur tersebut, misalnya urutan perubahan suara dari waktu ke waktu. Kombinasi ini menghasilkan model yang kuat untuk klasifikasi sinyal audio, analisis video, dan tugas-tugas lain yang menggabungkan dimensi spasial dan temporal.</p>
               </div>
                """,
                unsafe_allow_html=True
            )

        with col1:
            st.markdown(
                """
                <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                    <p><strong style="font-size: 1.4em;">Ilustrasi Algoritma CNN (Convolutional Neural Network)</strong></p>
                    <p style="text-align: justify;">
                    CNN (Convolutional Neural Network) adalah tipe jaringan saraf yang dirancang khusus untuk mengolah data berbentuk grid atau spatial data, seperti gambar. CNN mampu mengenali pola visual seperti tepi, tekstur, dan bentuk dengan cara yang sangat mirip dengan bagaimana otak manusia mengidentifikasi objek. CNN bekerja dengan menggunakan Convolutional Layer yaitu mengekstrak fitur dari data input yang digunakan, Pooling Layer yaitu lapisan pooling yang berfungsi untuk mengurangi dimensi data dan Fully Connected Layer yang bertindak sebagai klasifikasi untuk memetakan fitur yang ditemukan ke output akhir.
                </p>
                <div style="text-align: center; margin-top: 15px;">
                    <!-- Mengubah ukuran gambar dengan width 70% dan height auto -->
                    <img src="data:image/png;base64,{}" style="width: 70%; height: auto;">
                </div>
            </div>
            """.format(get_base64_of_bin_file("ILUSTRASI CNN.png")),
                unsafe_allow_html=True
            )


        with col2:
            st.markdown(
                """
                <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                <p><strong style="font-size: 1.4em;">Ilustrasi Algoritma GRU Gated Recurrent Unit</strong></p>
                <p style="text-align: justify;">
                        Gated Recurrent Unit (GRU) adalah tipe dari Recurrent Neural Network (RNN) yang dirancang untuk menangani data sekuensial dan memiliki mekanisme gate (pintu) yang mengontrol aliran informasi, sehingga membantu model untuk menangani dependensi temporal (seperti urutan waktu) dengan lebih baik. GRU bekerja dengan menggunakan mekanisme 2 gate yaitu Update Gate dan Reset Gate. Reset Gate mengatur seberapa banyak informasi dari waktu sebelumnya yang perlu dihapus sedangkan Update Gate menentukan seberapa banyak informasi dari waktu sebelumnya yang perlu disimpan.    </p>
                </p>
                <div style="text-align: center; margin-top: 15px;">
                    <!-- Mengubah ukuran gambar dengan width 70% dan height auto -->
                    <img src="data:image/png;base64,{}" style="width: 70%; height: auto;">
                </div>
            </div>
            """.format(get_base64_of_bin_file("ILUSTRASI GRU.png")),
                unsafe_allow_html=True
            )

        st.markdown(f'<div class="custom-header">Alur Pengembangan Model dan Penggunaan Website</div>', unsafe_allow_html=True)

        # Frame pertama (di atas)
        st.markdown(
            """
            <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                <p><strong style="font-size: 1.4em;">Flow Modelling</strong></p>
                <p style="text-align: justify;">
                    Gambar dibawah merupakan flow dari modelling menggunakan algoritma CNN yang dikombinasikan dengan GRU. Modelling dibagi kedalam 3 bagian yaitu pre-processing, modelling dan evaluasi. Pre-processing terdiri dari baca dan ekstrak data yang digunakan, augmentasi data yang berguna untuk meningkatkan jumlah dan variasi data pelatihan tanpa memerlukan pengumpulan data tambahan, ekstraksi fitur MFCC (Mel-Frequency Cepstral Coefficients) untuk mengonversi sinyal suara menjadi fitur yang lebih terstruktur yang dapat digunakan oleh algoritma dan encoding data atau transformasi tipe data menjadi bentuk yang disesuaikan oleh model yang digunakan dan langkah terakhir yaitu membagi data menjadi data pelatihan, pengujian dan validasi.
                </p>
                <p style="text-align: justify;">
                    Langkah selanjutnya yaitu membangun arsitektur model dengan menambah layer-layer yang dibutuhkan. Setelah itu data pelatihan dimasukkan dalam model yang sudah dibangun. Langkah terakhir yaitu evaluasi model untuk menilai seberapa baik model dalam sisi akurasi, presisi dan recall. Apabila model sudah dianggap baik, maka model tersebut diunduh serta dikembangkan ke dalam website.
                </p>
                <div style="text-align: center; margin-top: 15px;">
                    <!-- Mengubah ukuran gambar dengan width 70% dan height auto -->
                    <img src="data:image/png;base64,{}" style="width: 70%; height: auto;">
                </div>
            </div>
            """.format(get_base64_of_bin_file("Flow Modelling.png")),
            unsafe_allow_html=True
        )

        # Frame kedua (di bawah)
        st.markdown(
            """
            <div style="border: 1px solid #11ccf5; padding: 10px; margin-bottom: 10px; font-size: 1.2em; color: #333;">
                <p><strong style="font-size: 1.4em;">User Activity Flow</strong></p>
                <p style="text-align: justify;">
                    Gambar dibawah merupakan flow user activity yang bisa dilakukan oleh user. Pada gambar terlihat bahwa setiap pole merepresentasikan setiap Tab yang ada pada website ini. Gambar ini dapat memudahkan user dalam melihat secara keseluruhan apa saja fitur-fitur yang ada pada web ini.
                </p>
                <div style="text-align: center; margin-top: 15px;">
                    <!-- Mengubah ukuran gambar dengan width 60% dan height auto -->
                    <img src="data:image/png;base64,{}" style="width: 70%; height: auto;">
                </div>
            </div>
            """.format(get_base64_of_bin_file("User Activity.png")),
            unsafe_allow_html=True
        )


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: top;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

# Footer section
footer_html = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #d0d5d6;
        color: #212529;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #eaeaea;
    }
    </style>
    <div class="footer">
        <p>© 2024 Miko Ardian. All rights reserved. | <a href="https://your-website.com" target="_blank">Privacy Policy</a></p>
    </div>
    """

st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
