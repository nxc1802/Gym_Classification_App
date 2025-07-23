document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('results');
    const errorSection = document.getElementById('error');
    const videoInput = document.getElementById('video');
    const youtubeInput = document.getElementById('youtube_url');

    // Exercise names for display
    const ACTIONS = [
        "barbell biceps curl", "lateral raise", "push-up", "bench press",
        "chest fly machine", "deadlift", "decline bench press", "hammer curl",
        "hip thrust", "incline bench press", "lat pulldown", "leg extension",
        "leg raises", "plank", "pull Up", "romanian deadlift", "russian twist",
        "shoulder press", "squat", "t bar row", "tricep Pushdown", "tricep dips"
    ];

    // Clear other input when one is selected
    videoInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            youtubeInput.value = '';
        }
    });

    youtubeInput.addEventListener('input', function() {
        if (this.value.trim()) {
            videoInput.value = '';
        }
    });

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate inputs
        const hasFile = videoInput.files.length > 0;
        const hasYouTube = youtubeInput.value.trim();
        
        if (!hasFile && !hasYouTube) {
            showError('Please select a video file or provide a YouTube URL.');
            return;
        }

        // Show loading state
        setLoadingState(true);
        hideResults();
        hideError();

        try {
            const formData = new FormData();
            
            if (hasFile) {
                formData.append('video', videoInput.files[0]);
            }
            
            if (hasYouTube) {
                formData.append('youtube_url', youtubeInput.value.trim());
            }

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                displayResults(data.results);
            } else {
                showError(data.error || 'An error occurred during processing.');
            }
        } catch (error) {
            console.error('Upload error:', error);
            showError('Network error occurred. Please try again.');
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(loading) {
        submitBtn.disabled = loading;
        if (loading) {
            btnText.textContent = 'Processing...';
            loadingSpinner.style.display = 'inline-block';
        } else {
            btnText.textContent = 'Analyze Video';
            loadingSpinner.style.display = 'none';
        }
    }

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        errorSection.style.display = 'block';
        resultsSection.style.display = 'none';
    }

    function hideError() {
        errorSection.style.display = 'none';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    function displayResults(results) {
        console.log('Results:', results);
        
        // Display ensemble results
        displayPredictions('ensembleResults', results.ensemble_predictions);
        
        // Display individual model results
        displayPredictions('stgcnResults', results.individual_predictions.stgcn);
        displayPredictions('transformerResults', results.individual_predictions.transformer_12rel);
        displayPredictions('angleResults', results.individual_predictions.transformer_angle);
        displayPredictions('swin3dResults', results.individual_predictions.swin3d);
        
        // Display processing times
        document.getElementById('stgcnTime').textContent = results.processing_times.stgcn.toFixed(2);
        document.getElementById('transformerTime').textContent = results.processing_times.transformer_12rel.toFixed(2);
        document.getElementById('angleTime').textContent = results.processing_times.transformer_angle.toFixed(2);
        document.getElementById('swin3dTime').textContent = results.processing_times.swin3d.toFixed(2);
        
        // Display summary
        document.getElementById('totalTime').textContent = results.total_processing_time;
        document.getElementById('windowsProcessed').textContent = results.windows_processed || 'N/A';
        document.getElementById('mediapipeTime').textContent = results.processing_times.mediapipe.toFixed(2);
        
        // Show results
        resultsSection.style.display = 'block';
        hideError();
    }

    function displayPredictions(containerId, predictions) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        predictions.forEach((pred, index) => {
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            
            const exerciseName = document.createElement('span');
            exerciseName.className = 'exercise-name';
            exerciseName.textContent = `${index + 1}. ${pred.exercise}`;
            
            const probability = document.createElement('span');
            probability.className = 'probability';
            probability.textContent = `${(pred.probability * 100).toFixed(1)}%`;
            
            const probabilityBar = document.createElement('div');
            probabilityBar.className = 'probability-bar';
            
            const probabilityFill = document.createElement('div');
            probabilityFill.className = 'probability-fill';
            probabilityFill.style.width = `${pred.probability * 100}%`;
            
            probabilityBar.appendChild(probabilityFill);
            
            predItem.appendChild(exerciseName);
            predItem.appendChild(probability);
            
            const predContainer = document.createElement('div');
            predContainer.appendChild(predItem);
            predContainer.appendChild(probabilityBar);
            
            container.appendChild(predContainer);
        });
    }
}); 