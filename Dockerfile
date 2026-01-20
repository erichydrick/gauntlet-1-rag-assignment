FROM golang:1.25.6-alpine AS build

WORKDIR /app
ENV CGO_ENABLED=1

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o main main.go

FROM alpine:3.22.2 AS executor

WORKDIR /app

COPY --from=build /app/main /app/main

# The source documents to ingest
COPY documents documents

# The vector database
COPY vectors.db vectors.db
COPY vectors.db-wal vectors.db-wal

CMD ["./main"]
