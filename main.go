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

	fmt.Println("Creating storage schemas...")
	err := createTables()
	if err != nil {
		panic(fmt.Errorf("could not create the tables to use for the RAG pipeline: %v", err))
	}

	fmt.Println("Ingesting data...")
	err = ingestData()
	if err != nil {
		panic(fmt.Errorf("could not ingest documents: %v", err))
	}

	fmt.Printf("\n==================================================\n")
	fmt.Println("Starting application...")
	fmt.Printf("\n==================================================\n")
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

		processQuestion(question, false)

	}
	fmt.Printf("\n==================================================\n\n")
	fmt.Println("Application done...")
	fmt.Printf("\n==================================================\n")

	// TODO: RUN EVALS
	fmt.Printf("\n==================================================\n\n")
	fmt.Println("Starting evaluations...")
	fmt.Printf("\n==================================================\n")
	processQuestion("What version of Java are we upgrading to?", true)
	processQuestion("What is the goal of the booking overhaul?", true)
	processQuestion("What do we use for SMS?", true)
	processQuestion("How do providers integrate with Outlook?", true)
	processQuestion("How do I connect to Elastic Beanstalk?", true)

	fmt.Printf("\n\n\nShutting down...\n")
	closeDatabaseConnection()
}

func processQuestion(question string, eval bool) {

	/* Go ahead and define the output object so we can update it as we go */
	var err error
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
		fmt.Println("No filters - use all docs")
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

	res, err := queryForSimilarChunks(ctx, queryEmbeddings, whereClause, 5)
	if err != nil {
		panic(fmt.Errorf("could not find similar chunks: %v", err))
	}

	/*
		Checking the database for additional context drew a blank, return with the
		"nothing found" message
	*/
	if len(res) <= 0 {
		fmt.Println(userResponse)
		return
	}

	/*
		Evaluations get processed by a different model, with different instrcutions
	*/
	if eval {

		fmt.Println("*****Evaluating retrieved documents for", question, "*****")
		for _, hit := range res {

			evalRes, err := evaluateRetrieval(ctx, hit.content, question)
			if err != nil {
				panic(fmt.Errorf("error evaluating question %s: %v", question, err))
			}

			fmt.Println(hit.filename, "snippet -", evalRes)

		}
		fmt.Printf("****Finished evaluating documents for%s*****\n\n", question)
		return

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

func (o output) String() string {

	return fmt.Sprintf("{\n\t\"answer\": \"%s\",\n\t\"sources\": \"%s\",\n\t\"filters\": \"%s\"\n}", o.answer, strings.Join(o.sources, ", "), o.filters)

}
