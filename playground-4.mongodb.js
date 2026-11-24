/* === PLAYGROUND TAS AI STREAMLIT + MONGODB === */

// 1. Lihat semua database
db.getMongo().getDBs()

// 2. Pilih database TAS
use('test_db')

// 3. Lihat semua collection di test_db
db.getCollectionNames()

// 4. Cek total dokumen di iris_db
db.iris_db.countDocuments()

// 5. Tampilkan 5 dokumen pertama
db.iris_db.find().limit(5)
// === BUAT DATA IRIS ===
use('test_db')

db.iris_db.insertMany([
  {
    sepal_length: 5.1,
    sepal_width: 3.5,
    petal_length: 1.4,
    petal_width: 0.2,
    species: "setosa",
    timestamp: new Date()
  },
  {
    sepal_length: 7.0,
    sepal_width: 3.2,
    petal_length: 4.7,
    petal_width: 1.4,
    species: "versicolor",
    timestamp: new Date()
  },
  {
    sepal_length: 6.3,
    sepal_width: 3.3,
    petal_length: 6.0,
    petal_width: 2.5,
    species: "virginica",
    timestamp: new Date()
  }
])

// Verifikasi
db.iris_db.countDocuments()
db.iris_db.find()
// === VERIFIKASI DATA ===
db.iris_db.countDocuments()
db.iris_db.find()