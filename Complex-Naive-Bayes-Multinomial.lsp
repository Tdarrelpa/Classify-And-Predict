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
    "
    Combines multiple CSV files from a folder into a single output CSV file.
    Headers are written only once.
    "
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
    "
    Cleans text by lowercasing, removing numbers, and punctuation.
    "
  (let ((text (string-downcase text)))
    (setf text (remove-if #'digit-char-p text))
    (setf text (remove-if #'(lambda (c) (find c '("!" "@" "#" "$" "%" "^" "&" "*" "(" ")" "-" "_" "=" "+" "{" "[" "}" "]" ";" ":" 
    "'" "\"" "<" ">" "," "." "/" "?" "~" "`" "\\" "|")) :test #'equal) text))
    text))

; Membaca dataset
(defun read-dataset (file-path)
    "
    Reads a dataset from a CSV file, skipping the header.
    Assumes the first column is text and the second is an integer class.
    "
  ; Konteks manajer Common Lisp untuk membaca dataset setelah digabungkan
  (with-open-file (f file-path :direction :input :element-type 'character)
    (let ((reader (make-instance cl-csv:csv-reader :input-stream f)))
      (loop for row = (cl-csv:read-row reader) while row collect (cl-tuples:make-tuple (first row) (second row))))))

; Menghitung frekuensi kata
(defun word-frequencies (text)
    "
    Calculates word frequencies for a given text.
    "
  (let ((words (split-sequence:split-sequence #\Space text))
        (tbl (make-hash-table :test 'equal)))
    (loop for word in words
          do (incf (gethash word tbl 0)))
    tbl))

; Menghitung probabilitas kata
(defun calculate-word-probabilities (word-freqs total-words vocabulary-size)
  (let ((smoothing 1))
  (loop for (word freq) in word-freqs
        ; Before: collect (cons word (/ freq total-words))
        collect (cl-tuples:make-tuple word (log (or (/ (+ freq smoothing) (+ total-words (* vocabulary-size smoothing)))
        1e-10))))))

; Menghitung probabilitas kelas
(defun calculate-class-probabilities (class-freqs total-classes)
  (loop for (cls freq) in class-freqs
        ; Before: collect (cons cls (/ freq total-classes))
        collect (cl-tuples:make-tuple cls (log (or (/ freq total-classes) 1e-10)))))