FROM clfoundation/sbcl:2.2.2-slim

ENV VIRTUAL_ENV=/opt/venv

# Install build tools and git
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Clone cl-tuples, cl-csv, cl-ppcre, parse-number, and split-sequence into a fixed directory 
RUN mkdir -p /opt/lisp-libs \
    && git clone https://github.com/AccelerationNet/cl-csv /opt/lisp-libs/cl-csv \
    && git clone https://repo.or.cz/cl-tuples.git /opt/lisp-libs/cl-tuples \
    && git clone https://github.com/edicl/cl-ppcre /opt/lisp-libs/cl-ppcre \
    && git clone https://github.com/sharplispers/parse-number.git /opt/lisp-libs/parse-number \
    && git clone https://github.com/sharplispers/split-sequence.git /opt/lisp-libs/split-sequence

# Remove this block:
# RUN printf "(push #p\"/opt/lisp-libs/cl-csv/\" asdf:*central-registry*)\n" \
#     ... more lines ...
#     >> /etc/sbclrc

WORKDIR /app

# Copy your CL app source code
COPY app/ /app/

# Preload necessary systems and dump core for fast startup
RUN sbcl --non-interactive \
    --eval "(progn \
        (push #p\"/opt/lisp-libs/cl-csv/\" asdf:*central-registry*) \
        (push #p\"/opt/lisp-libs/cl-tuples/\" asdf:*central-registry*) \
        (push #p\"/opt/lisp-libs/cl-ppcre/\" asdf:*central-registry*) \
        (push #p\"/opt/lisp-libs/parse-number/\" asdf:*central-registry*) \
        (push #p\"/opt/lisp-libs/split-sequence/\" asdf:*central-registry*) \
        (asdf:load-system :cl-csv) \
        (asdf:load-system :cl-tuples) \
        (asdf:load-system :cl-ppcre) \
        (asdf:load-system :parse-number) \
        (asdf:load-system :split-sequence))"
    # --eval '(sb-ext:save-lisp-and-die "app.core" :executable t :toplevel #'(lambda () (in-package :x) (uiop:quit)))'

# Launch your saved core binary
# CMD ["./app.core"]
CMD ["sbcl", "--load", "Simple-Naive-Bayes-Multinomial.lsp"]

# Labeling image
LABEL maintainer="Timotius Darrel Putra Arma <darrel2208@gmail.com>"
LABEL version="1.0"
LABEL description="Simple (Built-in) and Complex (external) Naive Bayes Multinomial for classify and predict clickbait news"