package main

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/tursodatabase/go-libsql"
)

type vectorResult struct {
	content  string
	filename string
	score    float64
}

const (
	createDocumentsTable     = "CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, filename TEXT, content BLOB, embedding F64_BLOB(4096));"
	createMetadataTable      = "CREATE TABLE IF NOT EXISTS metadata (filename, topic TEXT, PRIMARY KEY (filename, topic));"
	addFilenameIndex         = "CREATE INDEX IF NOT EXISTS filename_idx ON documents(filename);"
	addTopicIndex            = "CREATE INDEX IF NOT EXISTS topic_idx ON metadata(topic);"
	dbPath                   = "file:./vectors.db"
	insertDocumentStatement  = "INSERT OR REPLACE INTO documents (filename, content, embedding) VALUES (?, ?, vector(?));"
	insertTopicsStatement    = "INSERT OR REPLACE INTO metadata (filename, topic) VALUES (?, ?);"
	lookupAllFiles           = "SELECT DISTINCT(filename) FROM metadata;"
	lookupMostRecentDocument = "SELECT id FROM documents ORDER BY id DESC LIMIT 1;"
	lookupFilesForTopics     = "SELECT DISTINCT(filename) FROM metadata WHERE topic IN (%s);"

	findSimilarChunks = `
		WITH vector_scores AS (
			SELECT filename,
				content,
				embedding, 
				1 - vector_distance_cos(embedding, vector(?)) AS similarity 
			FROM documents 
			%s
			ORDER BY similarity DESC
			LIMIT ?
		) 
		SELECT filename, content, similarity 
		FROM vector_scores
		WHERE similarity >= 0.5;
	`
)

var (
	db *sql.DB = nil
)

func closeDatabaseConnection() error {

	err := db.Close()
	if err != nil {
		return fmt.Errorf("could not close database connection: %v", err)
	}

	db = nil
	return nil

}

func connectToDatabase() error {

	if db != nil {
		return db.Ping()
	}

	var err error
	fmt.Println("DB at:", dbPath)
	db, err = sql.Open("libsql", dbPath)
	if err != nil {
		return fmt.Errorf("error connecting to database: %v", err)
	}

	return db.Ping()

}

func convertEmbeddings(rawEmbeddings []float64) string {

	stringifiedEmbeddings := make([]string, len(rawEmbeddings))

	for idx, rawValue := range rawEmbeddings {
		stringifiedEmbeddings[idx] = fmt.Sprintf("%f", rawValue)
	}

	return fmt.Sprintf("[%s]", strings.Join(stringifiedEmbeddings, ", "))

}

func createTables() error {

	/* Connect to the database if we haven't already */
	err := connectToDatabase()
	if err != nil {
		return fmt.Errorf("could not create database tables: %v", err)
	}

	/* Create the table schemas */
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("could not create transaction lock to create tables: %v", err)
	}
	_, err = db.Exec(createDocumentsTable)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("could not create the documents table: %v", err)
	}

	_, err = db.Exec(createMetadataTable)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("could not create the metadata table: %v", err)
	}

	_, err = db.Exec(addFilenameIndex)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("could not add the filename index to the document info: %v", err)
	}

	_, err = db.Exec(addTopicIndex)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("could not add the topic index to the metadata table: %v", err)
	}

	err = tx.Commit()
	if err != nil {
		return fmt.Errorf("could not commit transaction to create tables: %v", err)
	}
	return nil
}

func insertDocument(
	ctx context.Context,
	documentName string,
	chunks [][]byte,
	embeddings [][]float64,
	topics [][]string,
) error {

	if len(chunks) != len(embeddings) && len(chunks) != len(topics) {
		return fmt.Errorf("number of chunks, embeddings, and topic segments don't match (%d vs %d vs %d)", len(chunks), len(embeddings), len(topics))
	}

	for idx := range chunks {

		err := insertChunk(ctx, "documents/"+documentName, chunks[idx], embeddings[idx], topics[idx])
		if err != nil {
			return fmt.Errorf("could not insert chunk from %s: %v", documentName, err)
		}

	}

	return nil
}

func insertChunk(
	ctx context.Context,
	documentName string,
	text []byte,
	embeddings []float64,
	topics []string,
) error {

	fmt.Println("Inserting information for document", documentName)
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("could not start transaction to insert document data for %s: %v", documentName, err)
	}

	/* Save document name and embeddings */
	_, err = db.ExecContext(ctx, insertDocumentStatement, documentName, text, convertEmbeddings(embeddings))
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("could not insert document information for %s: %v", documentName, err)
	}

	var documentID int64
	err = db.QueryRowContext(ctx, lookupMostRecentDocument).Scan(&documentID)
	if err != nil {
		return fmt.Errorf("could not load most recent document id: %v", err)
	}

	/*
		Save document metadata
	*/
	for _, topicName := range topics {
		_, err := db.ExecContext(ctx, insertTopicsStatement, documentName, topicName)
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("could not insert topic information for document %s and topic %s: %v", documentName, topicName, err)
		}
	}

	fmt.Println("Committing")
	tx.Commit()
	fmt.Println("Wrote embeddings and metadata for", documentName)
	return nil

}

func queryForAllDocs(ctx context.Context) (filenames []string, resErr error) {

	filenames = []string{}
	resErr = nil

	rows, err := db.QueryContext(ctx, lookupAllFiles)
	if err != nil {
		resErr = fmt.Errorf("error looking up all filenames to use in querying: %v", err)
	}

	for rows.Next() {

		var document string
		err = rows.Scan(&document)
		if err != nil {
			resErr = fmt.Errorf("error reading document name from database: %v", err)
			return
		}

		filenames = append(filenames, document)

	}

	return

}

func queryForRelevantDocs(ctx context.Context, topics []string) (filenames []string, resErr error) {

	filenames = []string{}
	resErr = nil

	topicNames := ""
	for _, topic := range topics {
		topicNames += "'" + topic + "',"
	}
	topicNames = strings.TrimSuffix(topicNames, ",")
	fullQuery := fmt.Sprintf(lookupFilesForTopics, topicNames)

	rows, err := db.QueryContext(ctx, fullQuery)
	if err != nil {
		resErr = fmt.Errorf("error looking up documents with associated topics: %v", err)
		return
	}

	/* Read the (possibly) relevant filenames from the database */
	for rows.Next() {

		var name string
		if err = rows.Scan(&name); err != nil {
			resErr = fmt.Errorf("error reading the filename from the query results: %v", err)
		} else {
			filenames = append(filenames, name)
		}

	}
	if rows.Err() != nil {
		resErr = fmt.Errorf("error reading query results: %v", rows.Err())
		return
	}

	return

}

func queryForSimilarChunks(ctx context.Context, queryEmbedding []float64, whereClause string, limit int) (hits []vectorResult, resErr error) {

	resErr = nil

	fmt.Println("Searching for relevant chunks")

	rows, err := db.QueryContext(
		ctx,
		fmt.Sprintf(findSimilarChunks, whereClause),
		convertEmbeddings(queryEmbedding),
		limit,
	)
	if err != nil {
		resErr = fmt.Errorf("error looking up most relevant chunks: %v", err)
		return
	}

	hits = []vectorResult{}
	for rows.Next() {

		var record vectorResult
		err = rows.Scan(&record.filename, &record.content, &record.score)
		if err != nil {
			resErr = fmt.Errorf("error reading similarity results from database: %v", err)
			return
		}

		hits = append(hits, record)

	}

	return

}
