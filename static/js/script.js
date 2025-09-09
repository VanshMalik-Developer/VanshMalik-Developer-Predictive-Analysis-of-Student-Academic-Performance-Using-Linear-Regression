function fillDemo() {
    document.querySelector('[name="age"]').value = 17;
    document.querySelector('[name="gender"]').value = 'F';
    document.querySelector('[name="study_time"]').value = 3;
    document.querySelector('[name="failures"]').value = 0;
    document.querySelector('[name="school_support"]').value = 'yes';
    document.querySelector('[name="family_support"]').value = 'yes';
    document.querySelector('[name="paid_classes"]').value = 'yes';
    document.querySelector('[name="activities"]').value = 'yes';
    document.querySelector('[name="higher_ed"]').value = 'yes';
    document.querySelector('[name="internet"]').value = 'yes';
    document.querySelector('[name="absences"]').value = 2;
    document.querySelector('[name="G1"]').value = 15;
    document.querySelector('[name="G2"]').value = 16;
}

function validateForm() {
    let isValid = true;
    const fields = ['age', 'absences', 'G1', 'G2'];
    
    fields.forEach(field => {
        const input = document.querySelector(`[name="${field}"]`);
        const errorDiv = document.getElementById(`${field}-error`);
        const value = parseFloat(input.value);
        
        input.classList.remove('error-field');
        if (errorDiv) errorDiv.textContent = '';
        
        if (field === 'age' && (value < 15 || value > 25)) {
            input.classList.add('error-field');
            if (errorDiv) errorDiv.textContent = 'Age must be between 15 and 25';
            isValid = false;
        } else if (field === 'absences' && (value < 0 || value > 30)) {
            input.classList.add('error-field');
            if (errorDiv) errorDiv.textContent = 'Absences must be between 0 and 30';
            isValid = false;
        } else if ((field === 'G1' || field === 'G2') && (value < 0 || value > 20)) {
            input.classList.add('error-field');
            if (errorDiv) errorDiv.textContent = 'Grade must be between 0 and 20';
            isValid = false;
        }
    });
    
    return isValid;
}

function getPerformanceLevel(score) {
    if (score >= 16) return 'Excellent';
    else if (score >= 14) return 'Good';
    else if (score >= 12) return 'Average';
    else if (score >= 10) return 'Below Average';
    else return 'Poor';
}

function displayPredictionResult(data) {
    return `
        <div class="result success">
            <h3>🎯 Prediction Result</h3>
            <div class="prediction-details">
                <div class="prediction-card">
                    <h4>Final Grade</h4>
                    <div class="prediction-value">${data.prediction}/20</div>
                </div>
                <div class="prediction-card">
                    <h4>Letter Grade</h4>
                    <div class="prediction-value">${data.grade_letter}</div>
                </div>
                <div class="prediction-card">
                    <h4>Performance Level</h4>
                    <div class="prediction-value">${getPerformanceLevel(data.prediction)}</div>
                </div>
                <div class="prediction-card">
                    <h4>Confidence</h4>
                    <div class="prediction-value">${data.confidence || 'Medium'}</div>
                </div>
            </div>
            <p style="margin-top: 20px;"><em>Prediction based on Linear Regression model with 81% accuracy</em></p>
        </div>
    `;
}

// Initialize form submission handler when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!validateForm()) {
                return;
            }
            
            const predictBtn = document.getElementById('predictBtn');
            const originalText = predictBtn.innerHTML;
            predictBtn.innerHTML = '<span class="loading"></span> Predicting...';
            predictBtn.disabled = true;
            
            const formData = new FormData(this);
            const resultDiv = document.getElementById('result');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = displayPredictionResult(data);
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ Error</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result error">
                        <h3>❌ Error</h3>
                        <p>Failed to get prediction. Please try again.</p>
                    </div>
                `;
            } finally {
                predictBtn.innerHTML = originalText;
                predictBtn.disabled = false;
            }
        });
    }
});
