(defsystem "simple-naive-bayes"
  :version "0.1.0"
  :author "Timotius Darrel Putra Arma <darrel2208@gmail.com>"
  :license "CC BY-NC-ND 4.0"
  :description "A simple Multinomial Naive Bayes classifier for text categorization (e.g. clickbait detection)"
  :depends-on ("cl-csv"
               "cl-tuples"
               "cl-ppcre"
               "parse-number"
               "split-sequence")
  :components ((:file "Simple-Naive-Bayes-Multinomial")))
