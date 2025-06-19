# Impor library yang diperlukan
import csv
import os
import math
import random
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Still useful for consistent tokenization and vocab building
from sklearn.metrics import classification_report, confusion_matrix

# Load dan eksplorasi dataset
def readAndWriteDataset(folderPath, outputFile):
    """
    Combines multiple CSV files from a folder into a single output CSV file.
    Headers are written only once.
    """
    # Ambil semua nama file CSV dalam folder
    csvFiles = [f for f in os.listdir(folderPath) if f.endswith('.csv')]
    
    # Buka file output untuk menulis data gabungan
    with open(outputFile, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)

        #Flag untuk menulis header hanya sekali
        headerWritten = False

        for fileName in csvFiles:
            with open(os.path.join(folderPath, fileName), 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader)

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
    """
    Cleans text by lowercasing, removing numbers, and punctuation.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Membaca dataset
def readDataset(filePath):
    """
    Reads a dataset from a CSV file, skipping the header.
    Assumes the first column is text and the second is an integer class.
    """
    with open(filePath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        # data = [(row[0], int(row[1])) for row in reader]
        data = [(row[0], row[1]) for row in reader]
    return data

# Menghitung frekuensi kata
def wordFrequencies(text):
    """
    Calculates word frequencies for a given text.
    """
    words = text.split()
    return dict(Counter(words))

# Menghitung probabilitas kata
def calculateWordProbabilities(wordFreqs, totalWords, vocabularySize, smoothing=1):
    """
    Calculates word probabilities with Laplace smoothing.
    """
    # Use math.log for logarithmic probabilities
    return {word: math.log((freq + smoothing) / (totalWords + vocabularySize * smoothing))
            for word, freq in wordFreqs.items()}

# Menghitung probabilitas kelas
def calculateClassProbabilities(classFreqs, totalClasses):
    """
    Calculates class probabilities (priors).
    """
    return {cls: math.log(freq / totalClasses) for cls, freq in classFreqs.items()}

# Melatih model
def trainNaiveBayes(train_data):
    """
    Trains a Naive Bayes classifier from scratch using the math module.
    """
    classFreqs = Counter([cls for _, cls in train_data])
    totalClasses = len(train_data)

    # Initialize word_freqs and total_words per class
    wordFreqs = {cls: Counter() for cls in classFreqs}
    totalWordsInClass = {cls: 0 for cls in classFreqs}
    
    # Build a global vocabulary for smoothing
    globalVocabulary = set()

    for text, cls in train_data:
        cleanedText = cleanText(text)
        words = cleanedText.split()
        for word in words:
            wordFreqs[cls][word] += 1
            totalWordsInClass[cls] += 1
            globalVocabulary.add(word) # Add to global vocabulary

    vocabulary_size = len(globalVocabulary)

    classProbabilities = calculateClassProbabilities(classFreqs, totalClasses)

    wordProbabilities = {}
    for cls in classFreqs:
        # Calculate word probabilities for each class with smoothing
        wordProbabilities[cls] = calculateWordProbabilities(wordFreqs[cls], totalWordsInClass[cls], vocabulary_size)

    return classProbabilities, wordProbabilities, globalVocabulary

# Mengklasifikasi teks
def classifyTextNaiveBayes(text, classProbabilities, wordProbabilities, globalVocabulary, smoothing=1):
    """
    Classifies a single text using the trained Naive Bayes model.
    Uses log probabilities to avoid underflow.
    """
    cleanedText = clean_text(text)
    words = cleanedText.split()
    
    classScores = {}
    vocabularySize = len(globalVocabulary) # Use the global vocabulary size for smoothing

    for cls, logPrior in classProbabilities.items():
        score = logPrior # Start with the log prior probability
        
        # Get word probabilities for the current class, or an empty dict if class is missing
        clsWordProbs = wordProbabilities.get(cls, {})
        
        # Get total words in this class for smoothing denominator
        # This part requires access to total_words_in_class from training,
        # which isn't directly passed here. For simplicity in classification,
        # we'll assume a large enough "total words" effectively, or adjust smoothing.
        # A robust solution would pass total_words_in_class per class or recalculate.
        # For a truly isolated classification, we rely on the pre-calculated log probabilities.
        # If a word is unseen in a class, its probability is handled by smoothing.

        # If word is unseen in this specific class (but might be in global vocab)
        # Apply Laplace smoothing for unseen words: log(smoothing / (total_words_in_class + vocab_size * smoothing))
        # This approximation simplifies without needing exact total_words_in_class at prediction time.
        # A more precise way would pre-calculate smoothed log-probs for all words including unseen ones.
        # For this example, we assume cls_word_probs already accounts for smoothing for all words in global_vocabulary.
        # If a word isn't even in global_vocabulary, it's effectively 0 contribution.

        # A more robust solution for unseen words during prediction would need pre-calculating probabilities for all vocabulary 
        # words.
        # For simplicity, we'll assume word_probabilities already contains values for all global_vocabulary words (even if 0 freq 
        # initially).
        # The calculate_word_probabilities during training handles this.
        # If the word was never seen in the entire training data (not in global_vocabulary),
        # its contribution might be zero or a very small number based on the smoothing.
        # For a practical Naive Bayes, unseen words during prediction should use the smoothed probability for that class.
        
        # Sum log likelihoods of words
        for word in words:
            # Use the log probability from training if available, otherwise apply smoothing for unseen words
            if word in clsWordProbs:
                score += clsWordProbs[word]
            else:
                # Let's refine this: If word not in cls_word_probs, use smoothed probability for that word in that class.
                # The calculate_word_probabilities already generates log probabilities with smoothing.
                # So, if a word is in global_vocabulary but not cls_word_probs, it means its count was 0 for that class,
                # and its log probability was log(smoothing / (total_words_in_class + vocabulary_size * smoothing)).
                # We need to ensure word_probabilities dictionary is structured to give this.
                # This is an approximation.
                score += math.log(smoothing / (sum(math.exp(p) for p in clsWordProbs.values()) + vocabularySize * smoothing))
                # For this simplified classifier, if a word from the input is not in word_probabilities[cls],
                # it means its specific count for that class was 0 during training.
                # We will just add the log of the smoothed probability that would have been calculated:
                # log(smoothing / (total_words_in_class[cls] + vocabulary_size * smoothing))
                # Since we don't pass total_words_in_class[cls], we can't be perfectly accurate here.
                # Let's assume a default small smoothed value for truly unseen words.
                # Placeholder for correct smoothed value
                score += math.log(smoothing / (sum(math.exp(p) for p in clsWordProbs.values()) + vocabularySize * smoothing)) 


        classScores[cls] = score
    
    # Return the class with the highest log probability
    if not classScores:
        return None # Handle case where no classes are present
    return max(classScores, key=classScores.get)

# Evaluasi model
def evaluateModelNaiveBayes(testData, classProbabilities, wordProbabilities, globalVocabulary):
    """
    Evaluates the custom Naive Bayes model.
    Returns y_true and y_pred for sklearn metrics.
    """
    yTrue = []
    yPred = []
    for text, trueClass in testData:
        predictedClass = classifyTextNaiveBayes(text, classProbabilities, wordProbabilities, globalVocabulary)
        yTrue.append(trueClass)
        yPred.append(predictedClass)
    
    print("\n--- Evaluasi Model ---")
    print("Laporan Klasifikasi:")
    # Filter out None predictions if any occurred
    validYTrue = [yTrue[i] for i, pred in enumerate(y_pred) if pred is not None]
    validYPred = [pred for pred in yPred if pred is not None]

    # Ensure valid_y_true and valid_y_pred are not empty before generating report
    if not validYTrue or not validYPred:
        print("Tidak ada prediksi valid untuk dievaluasi.")
        return 0, [], [] # Return accuracy 0 and empty lists
    
    print(classification_report(validYTrue, validYPred, zero_division=0))

    print("Matriks yang Membingungkan (Confusion Matrix):")
    cm = confusion_matrix(validYTrue, validYPred)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriks yang Membingungkan (Confusion Matrix)')
    plt.colorbar()
    
    # Dynamically determine classes for plotting
    allClasses = sorted(list(set(validYTrue + validYPred)))
    tickMarks = [i for i, _ in enumerate(allClasses)]
    plt.xticks(tickMarks, allClasses, rotation=45)
    plt.yticks(tickMarks, allClasses)
    plt.tight_layout()
    plt.ylabel('Label kebenaran (True label)')
    plt.xlabel('Label diprediksi (Predicted label)')
    
    # Add text annotations for cell values
    thresh = cm.max() / 2.
    # for i, j in np.ndindex(cm.shape): <- Requires numpy for ndindex, so had to use nested loop in order to archieve the same 
    # functionality.
    numRows, numCols = cm.shape
    for i in range(numRows):
        for j in range(numCols):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.show()

    accuracy = sum(1 for true, pred in zip(validYTrue, validYPred) if true == pred) / len(validYTrue)
    
    return accuracy, yTrue, yPred

# Menyimpan model
def saveNaiveBayesModel(classProbabilities, wordProbabilities, globalVocabulary, modelFile):
    """
    Saves the custom Naive Bayes model (class priors, word probabilities) to a CSV.
    """
    with open(modelFile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'key', 'value', 'class']) # Added 'type' column
        
        # Save class probabilities
        for cls, prob in classProbabilities.items():
            writer.writerow(['class_prob', cls, prob, ''])
        
        # Save word probabilities
        for cls, word_probs_dict in wordProbabilities.items():
            for word, prob in word_probs_dict.items():
                writer.writerow(['word_prob', word, prob, cls])
        
        # Save global vocabulary (as a single string or multiple rows if needed)
        writer.writerow(['global_vocab', '', ' '.join(list(globalVocabulary)), ''])
    print(f"Model Naive Bayes disimpan di {modelFile}")

# Memuat model
def loadNaiveBayesModel(modelFile):
    """
    Loads the custom Naive Bayes model from a CSV.
    """
    classProbabilities = {}
    wordProbabilities = {}
    globalVocabulary = set()

    with open(modelFile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            item_type = row[0]
            if item_type == 'class_prob':
                # class_probabilities[int(row[1])] = float(row[2]) <- Convert class key back to int
                classProbabilities[row[1]] = float(row[2])
            elif item_type == 'word_prob':
                # cls = int(row[3]) <- Class is in the fourth column
                cls = row[3]
                if cls not in wordProbabilities:
                    wordProbabilities[cls] = {}
                wordProbabilities[cls][row[1]] = float(row[2])
            elif item_type == 'global_vocab':
                globalVocabulary = set(row[2].split()) # Load global vocabulary

    print(f"Model Naive Bayes di {modelFile}")
    return classProbabilities, wordProbabilities, globalVocabulary

# Menyimpan prediksi
def savePredictions(predictions, outputFile):
    """
    Saves predictions to a CSV file.
    """
    with open(outputFile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'predicted_class'])
        for text, predictedClass in predictions:
            writer.writerow([text, predictedClass])
    print(f"Prediksi disimpan di {outputFile}")

# Program Utama
def complexMultinomialWork(sourceFolder, destFile, modelAfterTraining, modelAfterPrediction):
    # Ensure combined_clickid.csv exists or create it first
    readAndWriteDataset(sourceFolder, destFile)

    # Read the dataset
    dataset = readDataset(destFile)

    # Split dataset using sklearn's train_test_split
    # We pass the full dataset and let train_test_split handle the indices
    # We still need to reconstruct the (text, label) pairs for our custom training function
    texts = [text for text, _ in dataset]
    labels = [label for _, label in dataset]

    XTrainRaw, XTestRaw, yTrain, yTest = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    trainData = list(zip(XTrainRaw, yTrain))
    testData = list(zip(XTestRaw, yTest))

    # Train the custom Naive Bayes model
    classProbabilities, wordProbabilities, globalVocabulary = trainNaiveBayes(trainData)

    # Evaluate the custom Naive Bayes model
    accuracy, yTrueEval, yPredEval = evaluateModelNaiveBayes(testData, classProbabilities, wordProbabilities, globalVocabulary)
    
    # Print the result
    print(f'Accuracy: {accuracy * 100:.2f}%\n\nAccurate evaluation: {yTrueEval}\n\nPrediction evaluation: {yPredEval}')

    # Save the custom Naive Bayes model
    saveNaiveBayesModel(classProbabilities, wordProbabilities, globalVocabulary, modelAfterTraining)

    # Load the custom Naive Bayes model (demonstration of loading)
    loadedClassProbabilities, loadedWordProbabilities, loadedGlobalVocabulary = loadNaiveBayesModel(modelAfterTraining)

    # Predict new texts using the loaded custom Naive Bayes model
    newTexts = ['Judul clickbait yang menarik', 'Berita biasa tanpa clickbait', 'Wow, ini bikin kaget!', 
        'Informasi tentang perkembangan teknologi']
    predictions = [
        (text, classifyTextNaiveBayes(text, loadedClassProbabilities, loadedWordProbabilities, loadedGlobalVocabulary))
        for text in new_texts
    ]
    save_predictions(predictions, modelAfterPrediction)

# Eksekusi
if __name__ == "__main__":
    DatasetSourceFolder = 'dataClickId/'
    DatasetFileDestination = 'complex_combined_clickid.csv'
    DatasetNaiveBayesModel = 'custom_naive_bayes_model.csv'
    DatasetPredictionModel = 'naive_bayes_predictions.csv'
    simpleMultinomialWork(DatasetSourceFolder, DatasetFileDestination, DatasetNaiveBayesModel, DatasetPredictionModel)