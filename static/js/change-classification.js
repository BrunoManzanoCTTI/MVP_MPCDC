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
    function displayResults(data, changeDetails) { // Added changeDetails parameter
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

            // Send the predicted label and change details to the chatbot
            if (window.sendChatbotMessage) {
                let detailsString = "Planned Change Details:\n";
                if (changeDetails) {
                    for (const key in changeDetails) {
                        // Only include fields that are relevant and have values
                        if (changeDetails[key] !== null && changeDetails[key] !== "" && key !== 'features') { 
                            // Exclude 'features' array if it was part of changeDetails for some reason
                            detailsString += `- ${key}: ${changeDetails[key]}\n`;
                        }
                    }
                } else {
                    detailsString += "- (No specific change details were passed to include here, check JavaScript logic)\n";
                }

                const chatbotMessage = `The following change has a predicted Priority of: ${data.predicted_label}.\n\n${detailsString}\nPlease analyze this change in the context of the provided cluster report and generate your structured JSON output (overall_explanation and two actionable_plans with confidence scores).`;
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
            cardContent += `<p><strong>Predicted Priority:</strong> ${data.predicted_label}</p>`;
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
        
        // Convert FormData to a plain object
        for (const [key, value] of formData.entries()) {
            // Convert change_request_status to integer if present
            if (key === 'change_request_status') {
                changeData[key] = parseInt(value, 10) || 0; // Default to 0 if parsing fails
            } else if (key === 'submit_date' || key === 'scheduled_start_date' || key === 'scheduled_end_date') {
                // Ensure date fields are included if they have values
                if (value) {
                    changeData[key] = value;
                } else {
                    // Send null if date is not provided, as per model requirements
                    changeData[key] = null; 
                }
            } else {
                changeData[key] = value;
            }
        }
        
        // Ensure all model features are present in changeData, setting to null if not provided by the form
        // This helps ensure the payload to the backend is consistent.
        const modelFeatures = [
            "submit_date", "scheduled_start_date", "scheduled_end_date", "f01_chr_serviceid",
            "serviceci", "ASORG", "ASGRP", "categorization_tier_1", "categorization_tier_2",
            "categorization_tier_3", "product_cat_tier_1", "product_cat_tier_2", "product_cat_tier_3",
            "change_request_status", "f01_chr_tipoafectacion"
            // "infrastructure_change_id" is also collected but primarily for identification/logging, not a direct model input feature.
        ];

        modelFeatures.forEach(feature => {
            if (!(feature in changeData)) {
                // If a feature expected by the model wasn't in the form or was empty, ensure it's set to null.
                // This handles cases like optional fields not being filled out.
                changeData[feature] = null; 
            } else if (changeData[feature] === "") {
                // Explicitly set empty strings to null for consistency, if backend expects null for empty optionals
                changeData[feature] = null;
            }
        });
        
        // infrastructure_change_id is handled by the loop above if present in FormData
        // If it's required but could be missing from the form, ensure it's handled:
        if (!('infrastructure_change_id' in changeData) || changeData['infrastructure_change_id'] === "") {
            // Decide if it should be null or if an error should be thrown if it's mandatory and missing
            // For now, let's assume it might be sent as empty and the backend handles it or it's truly optional for the API call itself.
            // If it's strictly required by the API and not just for model features, validation might be needed here or on the backend.
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
            
            // Display the results, passing changeData as well
            displayResults(results, changeData); 
            
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

    // Function to display actionable plans on the main page
    window.displayActionablePlans = function(plans) {
        const plansContainer = document.getElementById('actionablePlansContent');
        if (!plansContainer) {
            console.error("Actionable plans container not found.");
            return;
        }
        plansContainer.innerHTML = ''; // Clear previous plans

        if (plans && plans.length > 0) {
            const plansTitle = document.createElement('h5');
            plansTitle.textContent = 'Recommended Actionable Plans:';
            plansContainer.appendChild(plansTitle);

            const ul = document.createElement('ul');
            ul.className = 'actionable-plans-list'; // Add a class for styling

            plans.forEach(plan => {
                console.log("Processing plan object in displayActionablePlans:", JSON.stringify(plan)); // Log each plan object
                console.log("Processing plan object in displayActionablePlans:", JSON.stringify(plan)); // Log each plan object
                const li = document.createElement('li');
                li.className = 'actionable-plan-item'; // Add a class for styling

                const planDesc = document.createElement('p');
                // Correctly access plan.plan_description based on the actual key from LLM/console
                console.log("Value of plan.plan_description BEFORE marked.parseInline:", plan.plan_description); 
                planDesc.innerHTML = marked.parseInline(plan.plan_description || "No description provided."); 

                const planConfidence = document.createElement('p');
                planConfidence.className = 'plan-confidence';
                // Ensure consistency with confidence_score key from LLM output
                planConfidence.innerHTML = `<strong>Confidence:</strong> ${plan.confidence_score || 'N/A'}`;
                
                li.appendChild(planDesc);
                li.appendChild(planConfidence);
                ul.appendChild(li);
            });
            plansContainer.appendChild(ul);
            // Ensure the main classification results section is visible if it was hidden
            if (classificationResults.classList.contains('hidden')) {
                classificationResults.classList.remove('hidden');
            }
            // Scroll to the plans if they are added after the initial results
            plansContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            // Optionally, display a message if no plans are available
            // plansContainer.innerHTML = '<p>No specific actionable plans provided by the AI.</p>';
        }
    };
    
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
