package main

import (
	"context"
	"fmt"

	"github.com/jonathanhecl/gollama"
)

const (
	ragEvaluationTemplate = `
	You are a relevance judge. Determine if the retrieved document is relevant to answering the given question.

	A document is RELEVANT if it contains information that would help answer the question.
	A document is NOT RELEVANT if it contains no useful information for the question.

	Retrieved document:
	%s

	Question:
	%s

	Is this document relevant? Answer using "RELEVANT" or "NOT RELEVANT" followed by a brief explanation of why.
	`
	ragPromptTemplate = `
		You are a very helpful assistant with access to product documentation that includes designs, requirements, support information, and other information useful to developers working on medical practice management software. Your ONLY job is to look up information from these documents and use it to answer product requirement and support questions from developers.

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

func evaluateRetrieval(ctx context.Context, document string, question string) (judgement string, evalErr error) {

	judgement = ""
	evalErr = nil

	_, err := evaluatorLLM(ctx)
	if err != nil {
		evalErr = fmt.Errorf("could not initialize judge llm: %v", err)
		return
	}
	judge.SetTemperature(0).
		SetTopK(5)

	prompt := fmt.Sprintf(ragEvaluationTemplate, document, question)

	response, err := judge.Chat(ctx, prompt)
	if err != nil {
		evalErr = fmt.Errorf("error asking the llm: %v", err)
		return
	}

	judgement = response.Content
	return

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
