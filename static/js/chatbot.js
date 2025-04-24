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
                addMessage(data.response, false);
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
