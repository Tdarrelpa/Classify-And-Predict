# Impor library yang diperlukan
import csv
import os
import math
import random
import re
import string
from collections import Counter

# Load dan eksplorasi dataset
def readAndWriteDataset(folderPath, outputFile):
    # Tentukan folder yang berisi file CSV
    #folderPath = 'clickid_data/'
    
    # Ambil semua nama file CSV dalam folder
    csvFiles = [f for f in os.listdir(folderPath) if f.endswith('.csv')]
    
    # Tentukan nama file output
    # outputFile = 'combined_clickid.csv'
    
    # Buka file output untuk menulis data gabungan
    with open(outputFile, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        #Flag untuk menulis header hanya sekali
        headerWritten = False
        
        # Iterasi melalui setiap file CSV
        for fileName in csvFiles:
            with open(os.path.join(folderPath, fileName), 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader)  # Baca header
                
                # Tulis header jika belum ditulis
                if not headerWritten:
                    writer.writerow(header)
                    headerWritten = True
                    
                # Tulis data
                for row in reader:
                    writer.writerow(row)
    print(f"Semua file CSV telah digabungkan menjadi {outputFile}")

# Membersihkan teks
def cleanText(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Menghitung frekuensi kata¶
def wordFrequencies(text):
    words = text.split()
    return dict(Counter(words))

# Menghitung probabilitas kata
def calculateWordProbabilities(wordFreqs, totalWords):
    return {word: (freq / totalWords) for word, freq in wordFreqs.items()}

# Menghitung probabilitas kelas
def calculateClassProbabilities(classFreqs, totalClasses):
    return {cls: (freq / totalClasses) for cls, freq in classFreqs.items()}

# Menghitung probabilitas dokumen
def calculateDocumentProbability(text, classProbabilities, wordProbabilities):
    words = text.split()
    # Inisialiasi dengan classProbabilities biar nggak jadi variabel nganggur
    prob = {cls: math.log(classProbabilities[cls]) for cls in classProbabilities}
    for word in words:
        for cls in classProbabilities:
            # Gunakan get untuk menghindari KeyError jika 'kata'nya tidak ada
            wordProb = wordProbabilities.get(cls, {}).get(word, 1e-10)  # Gunakan nilai kecil untuk 'kata' yang tidak terlihat
            prob[cls] += math.log(wordProb)
    return prob

# Membaca dataset
def readDataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Lewati header
        # data = [(row[0], int(row[1])) for row in reader]
        data = [(row[0], row[1]) for row in reader]
    return data

# Membagi dataset menjadi training dan testing
def splitDataset(dataset, testSize=0.2):
    random.shuffle(dataset)
    splitIndex = int(len(dataset) * (1 - testSize))
    return dataset[:splitIndex], dataset[splitIndex:]

# Melatih model
def trainNaiveBayes(trainData):
    classFreqs = Counter([cls for _, cls in trainData])
    totalClasses = len(trainData)
    
    wordFreqs = {cls: Counter() for cls in classFreqs}
    totalWords = {cls: 0 for cls in classFreqs}
    
    for text, cls in trainData:
        cleanedText = cleanText(text)
        wordFreq = wordFrequencies(cleanedText)
        wordFreqs[cls].update(wordFreq)
        totalWords[cls] += len(wordFreq)
    
    classProbabilities = calculateClassProbabilities(classFreqs, totalClasses)
    wordProbabilities = {
        cls: calculateWordProbabilities(wordFreqs[cls], totalWords[cls])
        for cls in classFreqs
    }
    
    return classProbabilities, wordProbabilities

# Mengklasifikasi teks
def classifyText(text, classProbabilities, wordProbabilities):
    cleanedText = cleanText(text)
    classScores = calculateDocumentProbability(cleanedText, classProbabilities, wordProbabilities)
    return max(classScores, key=classScores.get)

# Evaluasi model
def evaluateModel(testData, classProbabilities, wordProbabilities):
    correct = 0
    for text, trueClass in testData:
        predictedClass = classifyText(text, classProbabilities, wordProbabilities)
        if predictedClass == trueClass:
            correct += 1
    accuracy = correct / len(testData)
    return accuracy

# Menyimpan model
def saveModel(classProbabilities, wordProbabilities, modelFile):
    with open(modelFile, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'probability'])
        for cls, prob in classProbabilities.items():
            writer.writerow([cls, prob])
        writer.writerow(['word', 'class', 'probability'])
        for cls, wordProbs in wordProbabilities.items():
            for word, prob in wordProbs.items():
                writer.writerow([word, cls, prob])
    print(f"Model Naive Bayes disimpan di {modelFile}")

# Memuat model
def loadModel(modelFile):
    classProbabilities = {}
    wordProbabilities = {}
    with open(modelFile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Lewati header
        for row in reader:
            if row[0] == 'word':
                break
            classProbabilities[row[0]] = float(row[1])
        for row in reader:
            if row[0] == 'word':
                continue
            if row[1] not in wordProbabilities:
                wordProbabilities[row[1]] = {}
            wordProbabilities[row[1]][row[0]] = float(row[2])
    print(f"Membaca model Naive Bayes di {modelFile}")
    return classProbabilities, wordProbabilities

# Menyimpan prediksi
def savePredictions(predictions, outputFile):
    with open(outputFile, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'predicted_class'])
        for text, predictedClass in predictions:
            writer.writerow([text, predictedClass])
    print(f"Prediksi disimpan di {outputFile}")

# Program utama
def simpleMultinomialWork(sourceFolder, destFile, modelAfterTraining, modelAfterPrediction):
    # Menggabungkan dataset, lalu lakukan operasi naive bayes
    readAndWriteDataset(sourceFolder, destFile)
    dataset = readDataset(destFile)
    trainData, testData = splitDataset(dataset)
    classProbabilities, wordProbabilities = trainNaiveBayes(trainData)
    accuracy = evaluateModel(testData, classProbabilities, wordProbabilities)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Simpan model
    saveModel(classProbabilities, wordProbabilities, modelAfterTraining)

    # Memuat model
    loadedClassProbabilities, loadedWordProbabilities = loadModel(modelAfterTraining)
    
    # Prediksi baru
    newTexts = ['Judul clickbait yang menarik', 'Berita biasa tanpa clickbait']
    predictions = [(text, classifyText(text, loadedClassProbabilities, loadedWordProbabilities)) for text in newTexts]
    savePredictions(predictions, modelAfterPrediction)

# Eksekusi
if __name__ == "__main__":
    DatasetSourceFolder = 'dataClickId/'
    DatasetFileDestination = 'combined_clickid.csv'
    DatasetNaiveBayesModel = 'naive_bayes_model.csv'
    DatasetPredictionModel = 'predictions.csv'
    simpleMultinomialWork(DatasetSourceFolder, DatasetFileDestination, DatasetNaiveBayesModel, DatasetPredictionModel)