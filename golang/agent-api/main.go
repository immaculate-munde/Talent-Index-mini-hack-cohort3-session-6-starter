// Local HTTP API for Mini Hack Assistant.
//
// Serves the Mini Hack Assistant UI and accepts POST /api/chat.
// The handler sends the user message through modelprovider (Gemini)
// and returns the agent's reply as JSON.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
	"mini-hack-cohort3-session6-golang/modelprovider"
)

const systemPrompt = "You are Mini Hack Assistant, a patient technical mentor for Team1 Kenya's Cohort 3 builders learning Avalanche and agentic apps. Keep answers under 150 words."

//go:embed all:web
var webFS embed.FS

type chatRequest struct {
	Message string `json:"message"`
}

type chatResponse struct {
	OK         bool   `json:"ok"`
	Provider   string `json:"provider"`
	Text       string `json:"text,omitempty"`
	StopReason string `json:"stopReason,omitempty"`
	Error      string `json:"error,omitempty"`
}

func chatHandler(client *modelprovider.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			_ = json.NewEncoder(w).Encode(chatResponse{OK: false, Error: "POST only"})
			return
		}

		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(chatResponse{OK: false, Error: "JSON body must include a non-empty message"})
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
		defer cancel()

		resp, err := client.GenerateText(ctx, systemPrompt, []modelprovider.Message{
			{Role: "user", Content: req.Message},
		}, nil)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(chatResponse{OK: false, Provider: client.Provider, Error: err.Error()})
			return
		}

		_ = json.NewEncoder(w).Encode(chatResponse{
			OK:         true,
			Provider:   client.Provider,
			Text:       resp.Text,
			StopReason: resp.StopReason,
		})
	}
}

func main() {
	_ = godotenv.Load()

	client, err := modelprovider.NewModelClient("")
	if err != nil {
		log.Fatal(err)
	}

	webRoot, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", chatHandler(client))
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.Handle("/", http.FileServer(http.FS(webRoot)))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	fmt.Printf("Agent UI:  http://localhost:%s\n", port)
	fmt.Printf("Chat API:  POST http://localhost:%s/api/chat\n", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
