# search engine IR NLP (TF IDF, WORD2VEC, FASTTEXT)
![alt text](image-4.png)
## run server in folder new

`uvicorn my_app:app --reload`

![alt text](image-3.png)

### PENJELASAN 
- jika belum train model, train dahulu dengan upload csv file yang ada di new folder "healthcare_dataset_with_clinic_notes.csv" atau file yang ada clininc notes
![alt text](image.png)
- jika sudah train semua dan ingin search pastikan reload model yang ingin di gunakan
![alt text](image-1.png)
- lakukan search seperti penyakit atau gejala atau obat
![alt text](image-2.png)
- NOTE: tidak perlu train ulang jika ingin search cukup load mode dengan endpoint load


## run frontend in folder frontend

`npm run dev`
