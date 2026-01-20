package main

import (
	"context"
	"fmt"

	"github.com/jonathanhecl/gollama"
)

const (
	ragPromptTemplate = `
		You are a very helpful assistant with access to product documentation that includes designs, requirements, and support information. Your ONLY job is to look up information from these documents and use it to answer questions from developers.

	You must use ONLY the information below to answer the developer's questions. If you don't have enough information to answer the question, tell the developer you don't know and what information you'd need to provide an answer. The developer can find this additional information offline and update the question later -- DO NOT GUESS.

	Context: 
	%s

	Question: 
	%s

	Answer:
	`
)

/*
embedder -> Does chunk embeddings
judge -> Handles evaluations
researcher -> Answers user questions
*/
var (
	embedder   *gollama.Gollama
	judge      *gollama.Gollama
	researcher *gollama.Gollama
)

func answerQuestion(ctx context.Context, promptContext string, question string) (answer string, llmErr error) {

	answer = ""
	llmErr = nil

	_, err := researcherLLM(ctx)
	if err != nil {
		llmErr = fmt.Errorf("could not initialize researcher llm: %v", err)
		return
	}
	researcher.SetTemperature(0).
		SetTopK(5)

	prompt := fmt.Sprintf(ragPromptTemplate, promptContext, question)

	response, err := researcher.Chat(ctx, prompt)
	if err != nil {
		llmErr = fmt.Errorf("error asking the llm: %v", err)
		return
	}

	answer = response.Content
	return

}

func embeddingLLM(ctx context.Context) (*gollama.Gollama, error) {

	if embedder == nil {
		embedder = gollama.New("nomic-embed-text")
		if err := embedder.PullIfMissing(ctx); err != nil {
			return nil, fmt.Errorf("could not load the nomic-embed-text model: %v", err)
		}
	}

	return embedder, nil

}

func evaluatorLLM(ctx context.Context) (*gollama.Gollama, error) {

	if judge == nil {
		judge = gollama.New("llama3.2")
		if err := judge.PullIfMissing(ctx); err != nil {
			return nil, fmt.Errorf("could not load the llama3.2 model: %v", err)
		}
	}
	return judge, nil

}

func researcherLLM(ctx context.Context) (*gollama.Gollama, error) {

	if researcher == nil {
		researcher = gollama.New("mistral")
		if err := researcher.PullIfMissing(ctx); err != nil {
			return nil, fmt.Errorf("could not load the mistral model: %v", err)
		}
	}

	return researcher, nil

}
