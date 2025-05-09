document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    // Function to add a message to the chat
    function addMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message user' : 'message bot';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (isUser) {
            // For user messages, use textContent to prevent XSS
            messageContent.textContent = message;
        } else {
            // For bot messages, parse Markdown and use innerHTML
            // Ensure marked.js is loaded in index.html
            if (typeof marked !== 'undefined') {
                messageContent.innerHTML = marked.parse(message);
            } else {
                // Fallback if marked.js is not loaded
                messageContent.textContent = message;
                console.error("marked.js not loaded. Cannot render Markdown.");
            }
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom of the chat with a slight delay to ensure content is rendered
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 10);
    }

    // Function to show loading indicator
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot loading';
        loadingDiv.id = 'loading-indicator';
        
        const loadingContent = document.createElement('div');
        loadingContent.className = 'message-content';
        
        // Add "Thinking..." text
        const thinkingText = document.createElement('span');
        thinkingText.textContent = 'Thinking';
        thinkingText.className = 'thinking-text';
        loadingContent.appendChild(thinkingText);

        // Create dots for animation
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dot.className = 'dot';
            loadingContent.appendChild(dot);
        }
        
        loadingDiv.appendChild(loadingContent);
        chatMessages.appendChild(loadingDiv);
        
        // Scroll to the bottom of the chat with a slight delay to ensure content is rendered
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 10);
    }

    // Function to remove loading indicator
    function removeLoading() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    // Function to send message to the server
    async function sendMessage(message) {
        try {
            showLoading();
            
            const response = await fetch('/mpcdc/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            removeLoading();
            
            if (data.error) {
                addMessage(`Error: ${data.error}`, false);
            } else {
                // Attempt to parse the response as JSON, with more robust extraction
                let llmResponse;
                let rawResponseText = data.response;

                try {
                    // First, try direct parsing
                    llmResponse = JSON.parse(rawResponseText);
                } catch (e) {
                    // If direct parsing fails, try to extract JSON from within markdown code blocks
                    console.warn("Direct JSON.parse failed. Attempting to extract JSON from markdown code block.", e);
                    const jsonRegex = /```json\s*([\s\S]*?)\s*```/; // Regex to find ```json ... ```
                    const match = rawResponseText.match(jsonRegex);
                    if (match && match[1]) {
                        try {
                            llmResponse = JSON.parse(match[1]);
                            console.info("Successfully extracted and parsed JSON from markdown code block.");
                        } catch (e2) {
                            console.error("Failed to parse extracted JSON:", e2);
                            llmResponse = null; // Ensure llmResponse is null if extraction parsing fails
                        }
                    } else {
                        llmResponse = null; // Ensure llmResponse is null if no JSON block found
                    }
                }

                if (llmResponse && llmResponse.overall_explanation) {
                    addMessage(llmResponse.overall_explanation, false); // Display explanation in chat

                    // Log the actionable_plans to inspect their structure
                    console.log("Parsed actionable_plans from LLM:", llmResponse.actionable_plans);

                    if (llmResponse.actionable_plans && window.displayActionablePlans) {
                        window.displayActionablePlans(llmResponse.actionable_plans);
                    } else {
                        // If plans are expected but missing in valid JSON, inform the user or log
                        console.warn("Actionable plans missing or displayActionablePlans function not available.");
                        addMessage("AI provided an explanation, but no actionable plans were found or could be displayed.", false);
                    }
                } else {
                    // If llmResponse is null or doesn't have the expected structure
                    console.error("Failed to parse LLM response as valid JSON or JSON structure is incorrect.");
                    console.error("Raw LLM response received:", rawResponseText);
                    // Display a user-friendly message and the raw response for debugging.
                    // The addMessage function will use marked.parse, so if the raw response is Markdown, it will render.
                    addMessage("The AI's response could not be fully processed into the expected format. Displaying raw response:\n\n" + rawResponseText, false);
                }
            }
        } catch (error) {
            removeLoading();
            addMessage(`Error: ${error.message}`, false);
        }
    }

    // Expose sendMessage function globally
    window.sendChatbotMessage = sendMessage;

    // Event listener for send button
    sendButton.addEventListener('click', function() {
        const message = userInput.value.trim();
        
        if (message) {
            addMessage(message, true);
            userInput.value = '';
            sendMessage(message);
        }
    });

    // Event listener for Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const message = userInput.value.trim();
            
            if (message) {
                addMessage(message, true);
                userInput.value = '';
                sendMessage(message);
            }
        }
    });
});
