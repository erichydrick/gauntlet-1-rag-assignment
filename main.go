package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
)

type output struct {
	answer  string
	filters []string
	sources []string
}

var (
	ctx context.Context
)

func main() {

	var cancel context.CancelFunc
	ctx, cancel = signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	err := createTables()
	if err != nil {
		panic(fmt.Errorf("could not create the tables to use for the RAG pipeline: %v", err))
	}

	err = ingestData()
	if err != nil {
		panic(fmt.Errorf("could not ingest documents: %v", err))
	}

	/* Go until the user quits, we'll run evals later */
	for true {

		/* Get the user question */
		stdInRead := bufio.NewReader(os.Stdin)
		fmt.Println("Please enter your question. If you're done, just type the phrase 'All done'")
		question, err := stdInRead.ReadString('\n')
		if err != nil {
			panic(fmt.Errorf("error reading user question: %v", err))
		}
		question = strings.TrimSpace(question)

		/* Check for the end phrase */
		if strings.ToLower(question) == "all done" {
			break
		}

		/* Go ahead and define the output object so we can update it as we go */
		var queryContext strings.Builder
		userResponse := output{
			answer:  "No answers to your question were found.",
			sources: []string{},
			filters: []string{},
		}

		whereClause := ""

		userResponse.filters = parseTopics([]byte(question))
		if len(userResponse.filters) > 0 {

			userResponse.sources, err = queryForRelevantDocs(ctx, userResponse.filters)
			if err != nil {
				panic(fmt.Errorf("could not find subset of documents to query: %v", err))
			}

			whereClause = "WHERE filename IN ("
			for _, doc := range userResponse.sources {
				whereClause += "'" + doc + "',"
			}
			whereClause = strings.TrimRight(whereClause, ",") + ")"

		} else {
			userResponse.sources, err = queryForAllDocs(ctx)
			if err != nil {
				panic(fmt.Errorf("could not find all documents for referencing: %v", err))
			}
		}

		fmt.Println("Getting the embedding model...")
		embedder, err := embeddingLLM(ctx)
		if err != nil {
			panic(fmt.Errorf("error loading the embedder model: %v", err))
		}
		fmt.Println("Getting the researcher model...")
		_, err = researcherLLM(ctx)
		if err != nil {
			panic(fmt.Errorf("error loading the researcher model: %v", err))
		}

		queryEmbeddings, err := embedder.Embedding(ctx, question)
		if err != nil {
			panic(fmt.Errorf("could not create embeddings for the query: %v", err))
		}

		res, err := queryForSimilarChunks(ctx, queryEmbeddings, whereClause, 10)
		if err != nil {
			panic(fmt.Errorf("could not find similar chunks: %v", err))
		}

		if len(res) <= 0 {
			fmt.Println(userResponse)
			continue
		}
		for _, hit := range res {
			fmt.Fprintf(&queryContext, "File: %s\nContent: %s\n", hit.filename, hit.content)
		}

		llmRes, err := answerQuestion(ctx, queryContext.String(), question)
		if err != nil {
			panic(fmt.Errorf("error answering the question: %v", err))
		}
		userResponse.answer = llmRes
		fmt.Println(userResponse)

	}

	// TODO: RUN EVALS

	fmt.Println("Shutting down...")
	closeDatabaseConnection()
}

func (o output) String() string {

	return fmt.Sprintf("{\n\t\"answer\": \"%s\",\n\t\"sources\": \"%s\",\n\t\"filters\": \"%s\"\n}", o.answer, strings.Join(o.sources, ", "), o.filters)

}
