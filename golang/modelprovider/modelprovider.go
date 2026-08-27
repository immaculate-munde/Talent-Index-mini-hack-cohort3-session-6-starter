// Package modelprovider is one factory, one interface, four providers.
// NewModelClient gives you back a Client with a Provider name and a
// GenerateText method. Every provider implements the same interface so
// your agent code never needs to know which model is actually running
// underneath.
//
// Provider selection: pass a name explicitly, NewModelClient("openai"),
// or pass "" and it reads MODEL_PROVIDER from the environment, falling
// back to "anthropic" if that is not set either.
//
// GenerateText always returns a Response with:
//
//	Text       the assistant's reply text ("" if it only called tools)
//	ToolCalls  a slice of ToolCall, normalized regardless of provider.
//	           Empty if the model did not call a tool, OR if this
//	           provider does not support tool calling yet.
//	StopReason the provider's own reason string, kept as is
//	Raw        the full untouched response, for provider-specific detail
//
// TOOL-CALLING SUPPORT, READ THIS BEFORE YOU BUILD AN AGENT
// Only the Anthropic client below implements tools. The other three
// accept a tools argument without erroring, but ignore it and always
// return an empty ToolCalls slice. Each provider's function-calling API
// shape is different enough that fully normalizing all four is real
// work, not a quick add. If your agent needs tools, build it on
// "anthropic" until the others catch up.
package modelprovider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go"
	openaioption "github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

var SupportedProviders = []string{"anthropic", "openai", "gemini", "ollama"}

// Message is the shape every provider's GenerateText accepts for
// conversation history: a role ("user" or "assistant") and content.
// For tool round trips, Content can also be the provider's own raw
// content block slice, see the Anthropic tool-calling example in
// chainkit-mcp-agent for how that is threaded through.
type Message struct {
	Role    string
	Content any
}

type ToolCall struct {
	ID    string
	Name  string
	Input map[string]any
}

type Response struct {
	Text       string
	ToolCalls  []ToolCall
	StopReason string
	Raw        any
}

// Tool matches Anthropic's tool definition shape: name, description,
// and a JSON Schema for the input. This is the shape currently
// supported by the Anthropic client; other providers ignore it.
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
}

type Client struct {
	Provider     string
	GenerateText func(ctx context.Context, systemPrompt string, messages []Message, tools []Tool) (*Response, error)
}

func getConfiguredProvider() (string, error) {
	provider := strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_PROVIDER")))
	if provider == "" {
		provider = "anthropic"
	}
	for _, p := range SupportedProviders {
		if p == provider {
			return provider, nil
		}
	}
	return "", fmt.Errorf("unsupported MODEL_PROVIDER %q, use one of: %s", provider, strings.Join(SupportedProviders, ", "))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func maxTokens() int64 {
	v := envOr("MAX_TOKENS", "1024")
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 1024
	}
	return n
}

func newAnthropicClient() (*Client, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY is not set")
	}

	client := anthropic.NewClient(anthropicoption.WithAPIKey(apiKey))
	model := envOr("ANTHROPIC_MODEL", "claude-sonnet-4-6")

	generateText := func(ctx context.Context, systemPrompt string, messages []Message, tools []Tool) (*Response, error) {
		anthropicMessages := make([]anthropic.MessageParam, 0, len(messages))
		for _, m := range messages {
			text, ok := m.Content.(string)
			if !ok {
				// Content is already provider-native (e.g. a tool round trip
				// carried through as anthropic.MessageParam). Callers that
				// need this should build the MessageParam slice themselves;
				// this simple path only covers plain text turns.
				return nil, fmt.Errorf("modelprovider: non-string message content requires building anthropic.MessageParam directly, see chainkit-mcp-agent for the pattern")
			}
			if m.Role == "assistant" {
				anthropicMessages = append(anthropicMessages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(text)))
			} else {
				anthropicMessages = append(anthropicMessages, anthropic.NewUserMessage(anthropic.NewTextBlock(text)))
			}
		}

		anthropicTools := make([]anthropic.ToolUnionParam, 0, len(tools))
		for _, t := range tools {
			schema := anthropic.ToolInputSchemaParam{Properties: t.InputSchema["properties"]}
			toolParam := anthropic.ToolUnionParamOfTool(schema, t.Name)
			anthropicTools = append(anthropicTools, toolParam)
		}

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: maxTokens(),
			System:    []anthropic.TextBlockParam{{Text: systemPrompt}},
			Messages:  anthropicMessages,
			Tools:     anthropicTools,
		})
		if err != nil {
			return nil, err
		}

		var text string
		var toolCalls []ToolCall
		for _, block := range resp.Content {
			switch block.Type {
			case "text":
				text = block.Text
			case "tool_use":
				var input map[string]any
				_ = json.Unmarshal(block.Input, &input)
				toolCalls = append(toolCalls, ToolCall{ID: block.ID, Name: block.Name, Input: input})
			}
		}

		return &Response{
			Text:       text,
			ToolCalls:  toolCalls,
			StopReason: string(resp.StopReason),
			Raw:        resp,
		}, nil
	}

	return &Client{Provider: "anthropic", GenerateText: generateText}, nil
}

func newOpenAIClient() (*Client, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY is not set")
	}

	client := openai.NewClient(openaioption.WithAPIKey(apiKey))
	model := envOr("OPENAI_MODEL", "gpt-4.1")

	generateText := func(ctx context.Context, systemPrompt string, messages []Message, tools []Tool) (*Response, error) {
		// Tool calling not yet implemented for this provider, see the
		// package doc comment. Plain text chat only, for now.
		chatMessages := []openai.ChatCompletionMessageParamUnion{openai.SystemMessage(systemPrompt)}
		for _, m := range messages {
			text, _ := m.Content.(string)
			if m.Role == "assistant" {
				chatMessages = append(chatMessages, openai.AssistantMessage(text))
			} else {
				chatMessages = append(chatMessages, openai.UserMessage(text))
			}
		}

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    shared.ChatModel(model),
			Messages: chatMessages,
		})
		if err != nil {
			return nil, err
		}

		choice := resp.Choices[0]
		return &Response{
			Text:       choice.Message.Content,
			ToolCalls:  nil,
			StopReason: choice.FinishReason,
			Raw:        resp,
		}, nil
	}

	return &Client{Provider: "openai", GenerateText: generateText}, nil
}

// newGeminiClient talks to the Gemini REST API directly rather than
// through google.golang.org/genai. That SDK exists, but this codebase
// could not compile against it in the environment this was written in
// (an outbound network restriction blocked google.golang.org). Plain
// REST avoids that dependency entirely. Verify the request/response
// shape against Gemini's current docs before relying on this in
// production, it was written from the documented shape, not tested
// against a live key.
func newGeminiClient() (*Client, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	model := envOr("GEMINI_MODEL", "gemini-3.6-flash")

	generateText := func(ctx context.Context, systemPrompt string, messages []Message, tools []Tool) (*Response, error) {
		type part struct {
			Text string `json:"text"`
		}
		type content struct {
			Role  string `json:"role"`
			Parts []part `json:"parts"`
		}
		type requestBody struct {
			SystemInstruction content   `json:"system_instruction"`
			Contents          []content `json:"contents"`
		}

		contents := make([]content, 0, len(messages))
		for _, m := range messages {
			text, _ := m.Content.(string)
			role := "user"
			if m.Role == "assistant" {
				role = "model"
			}
			contents = append(contents, content{Role: role, Parts: []part{{Text: text}}})
		}

		body := requestBody{
			SystemInstruction: content{Parts: []part{{Text: systemPrompt}}},
			Contents:          contents,
		}
		payload, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}

		url := fmt.Sprintf(
			"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
			model, apiKey,
		)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		rawBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("gemini request failed with status %d: %s", resp.StatusCode, string(rawBody))
		}

		var result struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
				FinishReason string `json:"finishReason"`
			} `json:"candidates"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rawBody, &result); err != nil {
			return nil, err
		}
		if result.Error != nil && result.Error.Message != "" {
			return nil, fmt.Errorf("gemini API error: %s", result.Error.Message)
		}

		text := ""
		stopReason := "unknown"
		if len(result.Candidates) > 0 {
			stopReason = result.Candidates[0].FinishReason
			if len(result.Candidates[0].Content.Parts) > 0 {
				text = result.Candidates[0].Content.Parts[0].Text
			}
		}

		return &Response{Text: text, ToolCalls: nil, StopReason: stopReason, Raw: result}, nil
	}

	return &Client{Provider: "gemini", GenerateText: generateText}, nil
}

func newOllamaClient() (*Client, error) {
	baseURL := envOr("OLLAMA_BASE_URL", "http://localhost:11434")
	model := envOr("OLLAMA_MODEL", "llama3.1")

	generateText := func(ctx context.Context, systemPrompt string, messages []Message, tools []Tool) (*Response, error) {
		// Tool calling not yet implemented for this provider, see the
		// package doc comment. Plain text chat only, for now.
		type chatMessage struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}
		type requestBody struct {
			Model    string        `json:"model"`
			Stream   bool          `json:"stream"`
			Messages []chatMessage `json:"messages"`
		}

		chatMessages := []chatMessage{{Role: "system", Content: systemPrompt}}
		for _, m := range messages {
			text, _ := m.Content.(string)
			chatMessages = append(chatMessages, chatMessage{Role: m.Role, Content: text})
		}

		payload, err := json.Marshal(requestBody{Model: model, Stream: false, Messages: chatMessages})
		if err != nil {
			return nil, err
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/chat", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf(
				"ollama request failed with status %d, is \"ollama serve\" running, and have you run \"ollama pull %s\"?",
				resp.StatusCode, model,
			)
		}

		var result struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			DoneReason string `json:"done_reason"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, err
		}

		return &Response{
			Text:       result.Message.Content,
			ToolCalls:  nil,
			StopReason: result.DoneReason,
			Raw:        result,
		}, nil
	}

	return &Client{Provider: "ollama", GenerateText: generateText}, nil
}

// NewModelClient resolves a provider by name, or by MODEL_PROVIDER in
// the environment if name is "".
func NewModelClient(name string) (*Client, error) {
	resolved := strings.ToLower(strings.TrimSpace(name))
	if resolved == "" {
		var err error
		resolved, err = getConfiguredProvider()
		if err != nil {
			return nil, err
		}
	}

	switch resolved {
	case "anthropic":
		return newAnthropicClient()
	case "openai":
		return newOpenAIClient()
	case "gemini":
		return newGeminiClient()
	case "ollama":
		return newOllamaClient()
	default:
		return nil, fmt.Errorf("unsupported provider: %s", resolved)
	}
}
