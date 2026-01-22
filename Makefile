all: delete-dbs run

build: vet
	go build -o main 

delete-dbs:
	rm vectors.db*

fmt:
	clear
	go fmt ./...

run: build
	./main

vet:
	go vet
