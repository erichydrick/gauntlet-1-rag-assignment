package main

import (
	"embed"
	"fmt"
	"io/fs"
	"math"
	"strings"
)

const chunkSize = 1000
const overlap = 200

var (
	//go:embed documents/*.txt
	documents embed.FS

	// TODO: CHANGE THESE
	topics map[string][]string = map[string][]string{
		"support": {
			"support", "ticket", "freshdesk", "zendesk", "superadmin",
		},
		"database": {
			"table", "database", "sql", "mysql",
		},
		"development": {
			"java", "javascript", "jquery", "jq", "class", "ci cd",
		},
		"testing": {
			"testing", "qa",
		},
		"analytics":    {"logi", "symphony", "metrics"},
		"security":     {"mfa", "security", "secure", "whitelist", "blacklist", "2fa", "permission", "encrypt", "decrypt", "login", "jwt"},
		"billing":      {"superbill", "financial", "payment", "invoice", "claim", "insurance"},
		"scheduling":   {"booking", "event", "appointment", "slot", "calendar", "availability"},
		"integrations": {"integration", "outlook", "bandwidth", "sms", "twillio"},
	}
)

func chunkText(fullText string) [][]byte {

	chunks := [][]byte{}

	for chunkStart := 0; chunkStart < len(fullText); chunkStart += (chunkSize - overlap) {

		/* Don't go past the end of the text */
		chunkEnd := int(math.Min(float64(len(fullText)), float64(chunkStart+chunkSize)))
		chunks = append(chunks, []byte(fullText[chunkStart:chunkEnd]))

	}

	return chunks
}

func ingestData() error {

	llm, err := embeddingLLM(ctx)
	if err != nil {
		return fmt.Errorf("could not load embedding model: %v", err)
	}

	err = fs.WalkDir(documents, "documents", func(path string, document fs.DirEntry, err error) error {

		if err != nil {
			return err
		}

		if document.IsDir() {
			fmt.Println("Skipping directory ", document.Name())
			return nil
		}
		fileContents, err := documents.ReadFile("documents/" + document.Name())
		if err != nil {
			return fmt.Errorf("could not read file %s: %v", document.Name(), err)
		}

		embeddings := [][]float64{}
		topics := [][]string{}

		documentText := string(fileContents)
		chunkedDoc := chunkText(documentText)

		/* We need per-chunk embeddings and metadata */
		for _, chunk := range chunkedDoc {

			topics = append(topics, parseTopics(chunk))

			if chunkEmbeddings, err := llm.Embedding(ctx, string(chunk)); err != nil {
				return fmt.Errorf("could not generate embeddings vector: %v", err)
			} else {
				embeddings = append(embeddings, chunkEmbeddings)
			}
		}

		if err = insertDocument(ctx, document.Name(), chunkedDoc, embeddings, topics); err != nil {
			return fmt.Errorf("could not insert data into database: %v", err)
		}
		return nil

	})

	return err
}

func parseTopics(chunkBytes []byte) []string {

	text := strings.ToLower(string(chunkBytes))
	topicsFound := []string{}

	for topicName, keywords := range topics {

		for _, keyword := range keywords {

			if strings.Contains(text, keyword) {
				topicsFound = append(topicsFound, topicName)
				break

			}

		}

	}

	return topicsFound

}
