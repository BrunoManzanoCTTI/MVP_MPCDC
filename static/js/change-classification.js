document.addEventListener('DOMContentLoaded', function() {
    const changeForm = document.getElementById('changeClassificationForm');
    const classificationResults = document.getElementById('classificationResults');
    const resultsContent = document.getElementById('resultsContent');
    const changeSection = document.getElementById('changeClassificationSection');
    
    // Add a click event to the chatbot message that mentions the form
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.addEventListener('click', function(e) {
        // Check if the clicked element or its parent has a message-content class
        const messageContent = e.target.closest('.message-content');
        if (messageContent && messageContent.textContent.includes('form below')) {
            scrollToChangeForm();
        }
    });
    
    // Function to format date as YYYY-MM-DDThh:mm:ss for datetime-local input fields
    function formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        const seconds = String(date.getSeconds()).padStart(2, '0');
        return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
    }

    // Function to scroll to the change form
    function scrollToChangeForm() {
        changeSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to show loading state
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'results-card';
        loadingDiv.id = 'loading-results';
        loadingDiv.innerHTML = '<p>Analyzing change data...</p>';
        
        resultsContent.innerHTML = '';
        resultsContent.appendChild(loadingDiv);
        classificationResults.classList.remove('hidden');
        
        // Scroll to results
        classificationResults.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to remove loading state
    function removeLoading() {
        const loadingIndicator = document.getElementById('loading-results');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }
    
    // Function to display results
    function displayResults(data) {
        removeLoading();
        
        // Clear previous results
        resultsContent.innerHTML = '';
        
        // Check for errors
        if (data.status === 'error') {
            const errorCard = document.createElement('div');
            errorCard.className = 'results-card error';
            errorCard.innerHTML = `
                <h5>Error</h5>
                <p>${data.message}</p>
                ${data.details ? `<p class="error-details">${data.details}</p>` : ''}
            `;
            resultsContent.appendChild(errorCard);
            return;
        }

        // Check for success status
        if (data.status === 'success' && data.predicted_label !== undefined) {
            // Create the result card using the new data structure
            const resultCard = createResultCard(data);
            resultsContent.appendChild(resultCard);

            // Send the predicted label to the chatbot
            if (window.sendChatbotMessage) {
                const chatbotMessage = `Based on the change details provided, the predicted impact is: ${data.predicted_label}. Can you provide more insights on this?`;
                window.sendChatbotMessage(chatbotMessage);

                // Enable chatbot input and button
                const userInput = document.getElementById('userInput');
                const sendButton = document.getElementById('sendButton');
                if (userInput) userInput.disabled = false;
                if (sendButton) sendButton.disabled = false;

            } else {
                console.error("Chatbot sendMessage function not available.");
            }

        } else {
            // Handle cases where status is success but prediction is missing, or other unexpected success formats
            const errorCard = document.createElement('div');
            errorCard.className = 'results-card error'; // Use error styling
            errorCard.innerHTML = `
                <h5>Prediction Unavailable</h5>
                <p>${data.message || 'The model returned a response, but the prediction could not be extracted.'}</p>
                ${data.raw_response ? `
                <div class="raw-response-toggle">
                    <a href="#" onclick="toggleRawResponse(event)">Show Technical Details</a>
                    <div class="raw-response hidden">
                        <pre>${JSON.stringify(data.raw_response, null, 2)}</pre>
                    </div>
                </div>` : ''}
            `;
            resultsContent.appendChild(errorCard);
        }
        
        // Scroll to results
        classificationResults.scrollIntoView({ behavior: 'smooth' });
    }

    // Function to create a result card based on the new API response
    function createResultCard(data) {
        const resultCard = document.createElement('div');
        resultCard.className = 'results-card success'; // Simple success styling

        // Build the HTML content based on the new response structure
        let cardContent = `<h5>Change Classification Result</h5>`;

        if (data.predicted_label !== undefined) {
            cardContent += `<p><strong>Predicted Type:</strong> ${data.predicted_label}</p>`;
        }

        if (data.raw_prediction !== undefined) {
            // Display raw prediction value for technical reference
            cardContent += `<p><small>Raw Prediction Value: ${data.raw_prediction.toFixed(4)}</small></p>`;
        }

        // Add raw response in a collapsible section if available (e.g., for debugging)
        // Note: The backend currently only includes raw_response on errors, but we keep the structure just in case
        if (data.raw_response) {
             cardContent += `
                <div class="raw-response-toggle">
                    <a href="#" onclick="toggleRawResponse(event)">Show Raw Model Response</a>
                    <div class="raw-response hidden">
                        <pre>${JSON.stringify(data.raw_response, null, 2)}</pre>
                    </div>
                </div>
            `;
        }

        resultCard.innerHTML = cardContent;
        return resultCard;
    }
    
    // Function to toggle raw response visibility
    window.toggleRawResponse = function(event) {
        event.preventDefault();
        const rawResponseDiv = event.target.nextElementSibling;
        const isHidden = rawResponseDiv.classList.contains('hidden');
        
        rawResponseDiv.classList.toggle('hidden');
        event.target.textContent = isHidden ? 'Hide Technical Details' : 'Show Technical Details';
    };
    
    // Handle form submission
    changeForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData(changeForm);
        const changeData = {};
        
        // Get date values for calculation
        const scheduledStartDate = formData.get('scheduled_start_date');
        const scheduledEndDate = formData.get('scheduled_end_date');

        // Convert FormData to a plain object, excluding the original date fields
        for (const [key, value] of formData.entries()) {
            // Exclude original date fields
            if (key !== 'scheduled_start_date' && key !== 'scheduled_end_date' && key !== 'submit_date') {
                 // Convert change_request_status to integer if present
                if (key === 'change_request_status') {
                    changeData[key] = parseInt(value, 10) || 0; // Default to 0 if parsing fails
                } else {
                    changeData[key] = value;
                }
            }
        }

        // Calculate change duration in minutes if both dates are provided
        if (scheduledStartDate && scheduledEndDate) {
            try {
                const start = new Date(scheduledStartDate);
                const end = new Date(scheduledEndDate);
                // Calculate difference in milliseconds, then convert to minutes
                const durationMs = end.getTime() - start.getTime();
                // Ensure duration is not negative
                const durationMinutes = Math.max(0, durationMs / (1000 * 60));
                changeData['change_duration'] = durationMinutes;
            } catch (e) {
                console.error("Error calculating date duration:", e);
                // Optionally add an error message to the user or set duration to a default/null
                changeData['change_duration'] = null; // Or 0, depending on desired behavior
            }
        } else {
             // Handle cases where dates are missing, set duration to null or 0
             changeData['change_duration'] = null; // Or 0
        }
        
        try {
            showLoading();
            
            // Send data to the API
            const response = await fetch('/mpcdc/classify_change', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(changeData)
            });
            
            const results = await response.json();
            
            // Display the results
            displayResults(results);
            
        } catch (error) {
            removeLoading();
            
            // Display error message
            resultsContent.innerHTML = `
                <div class="results-card">
                    <h5>Error</h5>
                    <p>An error occurred while processing your request: ${error.message}</p>
                </div>
            `;
            classificationResults.classList.remove('hidden');
        }
    });
    
    // Add a link in the initial chatbot message to scroll to the form
    const initialMessage = document.querySelector('.chatbot-messages .message.bot:first-child .message-content');
    if (initialMessage) {
        const formText = initialMessage.innerHTML.replace(
            'You can also use the form below',
            '<a href="#" class="scroll-to-form">You can also use the form below</a>'
        );
        initialMessage.innerHTML = formText;
        
        // Add click event for the link
        const formLink = initialMessage.querySelector('.scroll-to-form');
        if (formLink) {
            formLink.addEventListener('click', function(e) {
                e.preventDefault();
                scrollToChangeForm();
            });
        }
    }
});
