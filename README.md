# 🌍 Sun’iy yo‘ldosh tasvirlaridan foydalanib qishloq xo‘jaligi yerlarini chegaralarini aniqlash

<img width="293" height="172" alt="image" src="https://github.com/user-attachments/assets/661c08a7-90d5-49ad-a120-8845f4164076" />

---

## 📌 1. Loyihaning maqsadi
Ushbu loyiha **sun’iy yo‘ldosh tasvirlari** asosida qishloq xo‘jaligi yerlari chegaralarini aniqlash va ularni vizualizatsiya qilishga qaratilgan.  
Loyiha orqali:  
- Yer maydonlarini ajratish  
- Sun’iy yo‘ldosh tasvirlarini ishlash (preprocessing)  
- Modelni o‘qitish va baholash  
- Natijalarni vizual tarzda ko‘rsatish mumkin.  

---

## 📂 2. Loyiha tarkibi
Loyihada asosiy fayllar quyidagicha:

- `analysis.ipynb` – Ma’lumotlarni tahlil qilish va vizualizatsiya.  
- `main.py` – Asosiy pipeline jarayoni.  
- `train.py` – Modelni o‘rgatish kodi.  
- `model.py` – Model arxitekturasi va konfiguratsiyasi.  
- `metrics.py` – Baholash mezonlari (IoU, Dice va h.k.).  
- `draw.py` – Chegaralarni chizish va vizual natijalar.  
- `utils.py` – Yordamchi funksiyalar.  
- `test.py` – Modelni test qilish.  
- `requirements.txt` – Kutubxonalar ro‘yxati.  
- `files_Adam`, `files_Adam1`, `files_Adam2` – Model parametrlari yoki saqlangan fayllar.  
- `.idea/` – IDE konfiguratsiyasi.  

---

## ⚙️ 3. O‘rnatish (Installation)
1. Repozitoriyani klon qiling:
   ```bash
   git clone https://github.com/SardorNigmatov/Sun-iy-yo-ldosh-tasvirlaridan-foydalanib-qishloq-xo-jaligi-yerlarini-chegaralarini-aniqlash.git
   cd Sun-iy-yo-ldosh-tasvirlaridan-foydalanib-qishloq-xo-jaligi-yerlarini-chegaralarini-aniqlash
