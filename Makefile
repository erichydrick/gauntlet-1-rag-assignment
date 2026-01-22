all: run

fmt: 
    clear
    go fmt ./...

run: vet
    rm vectors.db*
    go build -o main 
    ./main

vet: fmt
    go vet ./...
