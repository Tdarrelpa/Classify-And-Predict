; Load the necessary library first
(asdf:load-system :cl-csv)
(asdf:load-system :cl-tuples)
(asdf:load-system :cl-ppcre)
(asdf:load-system :parse-number)
(asdf:load-system :split-sequence)

; Import necessary libraries
(defpackage :simple-naive-bayes
  (:use :cl :cl-csv :uiop :cl-ppcre :cl-tuples :parse-number :split-sequence))

(in-package :simple-naive-bayes)

(defun make-counter (&optional (sequence nil) &key (test #'eql))
  "Create a hash-table mapping each key in SEQUENCE to its count."
  (let ((counts (make-hash-table :test test)))
    (loop for x in sequence
          do (incf (gethash x counts 0)))
    counts))

(defun hash-table-keys (table)
  "Return a list of all keys in the hash TABLE."
  (declare (type hash-table table))
  (let (keys)
    (maphash (lambda (k _v)
               (push k keys))
             table)
    keys))

(defun shuffle-in-place (seq &key (start 0) (end (length seq)))
  "Shuffle a sequence (array or vector) in place between START (inclusive) and END (exclusive)."
  (declare (type (or vector simple-string adjustable-vector) seq)
           (integer start end))
  (let ((n end))
    (loop for i from (1- end) downto (1+ start) do
      (let* ((j (+ start (random (- i start 0))))
             (temp (elt seq i)))
        (setf (elt seq i) (elt seq j)
              (elt seq j) temp))))
  seq)


; Memuat dan mengeksplorasi dataset
(defun read-and-write-dataset (folder-path output-file)
  ; Ambil semua nama file CSV dalam folder
  (let ((csv-files (remove-if-not (lambda (f) (string= (subseq f (- (length f) 4)) ".csv"))
                                   (directory (merge-pathnames "*.csv" folder-path)))))
    ; Buka file output untuk menulis data gabungan
    (with-open-file (outfile output-file :direction :output :element-type 'character :if-exists :supersede)
      (let ((writer (make-instance cl-csv:csv-writer :output-stream outfile)))
        ; Flag untuk menulis header hanya sekali
        (let ((header-written nil))
          ; Iterasi melalui setiap file CSV
          (dolist (file-name csv-files)
            (with-open-file (infile (merge-pathnames file-name folder-path) :direction :input :element-type 'character)
              (let ((reader (make-instance cl-csv:csv-reader :input-stream infile)))
                (let ((header (cl-csv:read-row reader))) ; Baca header
                  ; Tulis header jika belum ditulis
                  (unless header-written
                    (cl-csv:write-row writer header)
                    (setf header-written t))
                  ; Tulis data
                  (loop for row = (cl-csv:read-row reader) while row do
                    (cl-csv:write-row writer row)))))))
      (format t "Semua file CSV telah digabungkan menjadi ~A~%" output-file)))))

; Membersihkan teks
(defun clean-text (text)
  (let ((text (string-downcase text)))
    (setf text (remove-if #'digit-char-p text))
    (setf text (remove-if #'(lambda (c) (find c '("!" "@" "#" "$" "%" "^" "&" "*" "(" ")" "-" "_" "=" "+" "{" "[" "}" "]" ";" ":" 
    "'" "\"" "<" ">" "," "." "/" "?" "~" "`" "\\" "|")) :test #'equal) text))
    text))

; Menghitung frekuensi kata
(defun word-frequencies (text)
  (let ((words (split-sequence:split-sequence #\Space text))
        (tbl (make-hash-table :test 'equal)))
    (loop for word in words
          do (incf (gethash word tbl 0)))
    tbl))

; Menghitung probabilitas kata
(defun calculate-word-probabilities (word-freqs total-words)
  (loop for (word freq) in word-freqs
        ; Before: collect (cons word (/ freq total-words))
        collect (cl-tuples:make-tuple word (/ freq total-words))))

; Menghitung probabilitas kelas
(defun calculate-class-probabilities (class-freqs total-classes)
  (loop for (cls freq) in class-freqs
        ; Before: collect (cons cls (/ freq total-classes))
        collect (cl-tuples:make-tuple cls (/ freq total-classes))))

; Menghitung probabilitas dokumen
(defun calculate-document-probability (text class-probabilities word-probabilities)
  (let ((words (split-sequence:split-sequence #\Space text))
        ; Inisialiasi dengan class-probabilities biar nggak jadi variabel nganggur
        (prob (make-hash-table)))
    (loop for cls in (hash-table-keys class-probabilities) do
      (setf (gethash cls prob) (log (gethash cls class-probabilities))))
    (loop for word in words do
      (loop for cls in (hash-table-keys class-probabilities) do
        ; Gunakan get untuk menghindari KeyError jika 'kata'nya tidak ada
        (let ((word-prob (gethash word (gethash cls word-probabilities)))) ; Gunakan nilai kecil untuk 'kata' yang tidak terlihat
          (setf (gethash cls prob) (+ (gethash cls prob) (log (or word-prob 1e-10))))))
    prob)))

; Membaca dataset
(defun read-dataset (file-path)
  ; Konteks manajer Common Lisp untuk membaca dataset setelah digabungkan
  (with-open-file (f file-path :direction :input :element-type 'character)
    (let ((reader (make-instance cl-csv:csv-reader :input-stream f)))
      (loop for row = (cl-csv:read-row reader) while row collect (cl-tuples:make-tuple (first row) (second row))))))

; Membagi dataset menjadi training dan testing
(defun split-dataset (dataset &optional (test-size 0.2))
  (let ((shuffled (shuffle-in-place dataset)))
    (let ((split-index (floor (* (length shuffled) (- 1 test-size)))))
    ; Before: (values (subseq shuffled 0 split-index) (subseq shuffled split-index))
      (cl-tuples:make-tuple (subseq shuffled 0 split-index) (subseq shuffled split-index)))))

; Train Naive Bayes model
(defun train-naive-bayes (train-data)
  (let ((class-freqs   (make-counter (mapcar #'second train-data)))
         (total-classes (length train-data))
         (word-freqs    (make-hash-table :test #'equal)) ; cls → word-counts-table
         (total-words   (make-hash-table :test #'equal))) ; cls → total word count
    (loop for cls in (hash-table-keys class-freqs) do
      (setf (maphash (lambda (cls _cnt)
                                     (setf (gethash cls word-freqs) 0))
                                   class-freqs)))
    (loop for cls in (hash-table-keys class-freqs) do
      (setf (gethash cls total-words) 0))
    (loop for (text cls) in train-data do
      (incf (gethash cls class-freqs 0))
      ; The existence of cleaned-text here was making word-freq uncolored, something along the line of Common Lisp can't read it
      ; That's why it's erased now, and the function call that defined cleaned-text is put inside another function call
      ; Before: (cleaned-text(clean-text text))
      (let ((word-freq (word-frequencies (clean-text text))))
        (incf (gethash cls total-words 0) (length word-freq))
        (loop for (word freq) in word-freq do
          (incf (gethash word (gethash cls word-freqs (make-hash-table)) 0) freq))))
    (let ((class-probabilities (calculate-class-probabilities class-freqs total-classes))
          (word-probabilities (make-hash-table :test #'equal)))
      (loop for cls in (hash-table-keys class-freqs) do
        (setf (gethash cls word-probabilities) (calculate-word-probabilities (gethash cls word-freqs) (gethash cls total-words))))
        ; Before:  (values class-probabilities word-probabilities)
      (cl-tuples:make-tuple class-probabilities word-probabilities))))

; Mengklasifikasi teks
(defun classify-text (text class-probabilities word-probabilities)
  ; The existence of cleaned-text here was making class-scores uncolored, something along the line of Common Lisp can't read it
  ; That's why it's erased now, and the function call that defined cleaned-text is put inside another function call
  ; Before: (cleaned-text(clean-text text))
  (let ((class-scores (calculate-document-probability (clean-text text) class-probabilities word-probabilities)))
    (reduce (lambda (a b) (if (> (gethash b class-scores) (gethash a class-scores)) b a)) (hash-table-keys class-scores))))

; Evaluasi model
(defun evaluate-model (test-data class-probabilities word-probabilities)
  (let ((correct 0))
    (loop for (text true-class) in test-data do
      (let ((predicted-class (classify-text text class-probabilities word-probabilities)))
        (when (string= predicted-class true-class)
          (incf correct))))
    (/ correct (length test-data))))

; Menyimpan model
(defun save-model (class-probabilities word-probabilities model-file)
  ; Konteks manajer Common Lisp untuk menulis hasil training Naive Bayes di model-file
  (with-open-file (f model-file :direction :output :element-type 'character)
    (let ((writer (make-instance cl-csv:csv-writer :output-stream f)))
      (cl-csv:write-row writer '("class" "probability"))
      (loop for (cls prob) in class-probabilities do
        (cl-csv:write-row writer (list cls prob)))
      (cl-csv:write-row writer '("word" "class" "probability"))
      (loop for cls in (hash-table-keys word-probabilities) do
        (loop for (word prob) in (gethash cls word-probabilities) do
          (cl-csv:write-row writer (list word cls prob)))))
    (format t "Model Naive Bayes disimpan di ~A~%" model-file)))

; Memuat model
(defun load-model (model-file)
  (let ((class-probabilities (make-hash-table))
        (word-probabilities (make-hash-table)))
    ; Konteks manajer Common Lisp untuk membaca model Naive Bayes dari model-file
    (with-open-file (f model-file :direction :input :element-type 'character)
      (let ((reader (make-instance cl-csv:csv-reader :input-stream f)))
        (loop for row = (cl-csv:read-row reader) while row do
          (if (string= (first row) "word")
              (return)
              (setf (gethash (first row) class-probabilities) (parse-float (second row)))))
        (loop for row = (cl-csv:read-row reader) while row do
          (if (string= (first row) "word")
              (continue)
              (let ((word (first row))
                    (cls (second row))
                    (prob (parse-number:parse-real-number (third row))))
                (setf (gethash cls (gethash cls word-probabilities (make-hash-table))) word prob))))))
    (format t "Membaca model Naive Bayes di ~A~%" model-file)
    ; Before: (values class-probabilities word-probabilities)
    (cl-tuples:make-tuple class-probabilities word-probabilities)))

; Menyimpan prediksi
(defun save-predictions (predictions output-file)
  ; Konteks manajer Common Lisp untuk menulis hasil prediksi di output-file
  (with-open-file (f output-file :direction :output :element-type 'character)
    (let ((writer (make-instance cl-csv:csv-writer :output-stream f)))
      (cl-csv:write-row writer '("text" "predicted_class"))
      (loop for (text predicted-class) in predictions do
        (cl-csv:write-row writer (list text predicted-class))))
    (format t "Prediksi disimpan di ~A~%" output-file)))

; Program utama
(defun simple-multinomial-work (source-folder dest-file model-after-training model-after-prediction)
  ; Menggabungkan dataset, lalu lakukan operasi naive bayes
  (read-and-write-dataset source-folder dest-file)
  (let ((dataset (read-dataset dest-file)))
    (multiple-value-bind (train-data test-data) (split-dataset dataset)
      (multiple-value-bind (class-probabilities word-probabilities) (train-naive-bayes train-data)
        (let ((accuracy (evaluate-model test-data class-probabilities word-probabilities)))
          (format t "Accuracy: ~A%~%" (* accuracy 100))
          ; Simpan model
          (save-model class-probabilities word-probabilities model-after-training)
          ; Memuat model
          (multiple-value-bind (loaded-class-probabilities loaded-word-probabilities) (load-model model-after-training)
            ; Prediksi baru
            (let ((new-texts '("Judul clickbait yang menarik" "Berita biasa tanpa clickbait")))
                ; Before: (cons text (classify-text text loaded-class-probabilities loaded-word-probabilities))
              (let ((predictions (loop for text in new-texts
                                       collect (cl-tuples:make-tuple text (classify-text text loaded-class-probabilities 
                                       loaded-word-probabilities)))))
                (save-predictions predictions model-after-prediction)))))))))

; Eksekusi
(defun main ()
  (let ((dataset-source-folder "dataClickId/")
        (dataset-file-destination "combined_clickid.csv")
        (dataset-naive-bayes-model "naive_bayes_model.csv")
        (dataset-prediction-model "predictions.csv"))
    (simple-multinomial-work dataset-source-folder dataset-file-destination dataset-naive-bayes-model dataset-prediction-model)))

; Jalankan fungsi main
(main)