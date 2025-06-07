/**
 * CardioInsight AI - Main JavaScript
 * 
 * This file contains the main JavaScript functions for the CardioInsight AI web application.
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('CardioInsight AI Web Application Loaded');
    
    // Initialize tooltips
    initTooltips();
    
    // Initialize popovers
    initPopovers();
    
    // Initialize ECG visualization if on the visualization page
    if (document.querySelector('.ecg-visualization')) {
        initECGVisualization();
    }
    
    // Initialize file upload preview if on the upload page
    if (document.querySelector('.file-upload')) {
        initFileUpload();
    }
    
    // Initialize analysis form if on the analysis page
    if (document.querySelector('.analysis-form')) {
        initAnalysisForm();
    }
    
    // Add fade-in animation to cards
    animateCards();
});

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize Bootstrap popovers
 */
function initPopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Initialize ECG visualization
 */
function initECGVisualization() {
    console.log('Initializing ECG visualization');
    
    // Get plot containers
    const plotContainers = document.querySelectorAll('.plot-container');
    
    // Initialize each plot
    plotContainers.forEach(container => {
        const plotData = JSON.parse(container.dataset.plot);
        Plotly.newPlot(container.id, plotData.data, plotData.layout);
    });
}

/**
 * Initialize file upload preview
 */
function initFileUpload() {
    console.log('Initializing file upload');
    
    const fileInput = document.querySelector('.file-upload input[type="file"]');
    const filePreview = document.querySelector('.file-preview');
    const fileNameDisplay = document.querySelector('.file-name');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file selected';
            fileNameDisplay.textContent = fileName;
            
            // Show file info if a file is selected
            if (e.target.files[0]) {
                const fileSize = (e.target.files[0].size / 1024).toFixed(2) + ' KB';
                const fileType = e.target.files[0].type || 'Unknown';
                
                filePreview.innerHTML = `
                    <div class="alert alert-info">
                        <strong>File:</strong> ${fileName}<br>
                        <strong>Size:</strong> ${fileSize}<br>
                        <strong>Type:</strong> ${fileType}
                    </div>
                `;
                
                // Show additional options based on file type
                if (fileType.includes('csv') || fileType.includes('text')) {
                    showCSVOptions();
                } else if (fileName.endsWith('.npy') || fileName.endsWith('.npz')) {
                    showNumpyOptions();
                } else if (fileName.endsWith('.mat')) {
                    showMatlabOptions();
                }
            } else {
                filePreview.innerHTML = '';
            }
        });
    }
}

/**
 * Show CSV file options
 */
function showCSVOptions() {
    const optionsContainer = document.querySelector('.file-options');
    if (optionsContainer) {
        optionsContainer.innerHTML = `
            <div class="card mt-3">
                <div class="card-header">CSV Options</div>
                <div class="card-body">
                    <div class="mb-3">
                        <label for="delimiter" class="form-label">Delimiter</label>
                        <select class="form-select" id="delimiter" name="delimiter">
                            <option value="comma" selected>Comma (,)</option>
                            <option value="semicolon">Semicolon (;)</option>
                            <option value="tab">Tab</option>
                            <option value="space">Space</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="header" class="form-label">Header Row</label>
                        <select class="form-select" id="header" name="header">
                            <option value="true" selected>Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="sampling_rate" class="form-label">Sampling Rate (Hz)</label>
                        <input type="number" class="form-control" id="sampling_rate" name="sampling_rate" value="250">
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Show NumPy file options
 */
function showNumpyOptions() {
    const optionsContainer = document.querySelector('.file-options');
    if (optionsContainer) {
        optionsContainer.innerHTML = `
            <div class="card mt-3">
                <div class="card-header">NumPy Options</div>
                <div class="card-body">
                    <div class="mb-3">
                        <label for="sampling_rate" class="form-label">Sampling Rate (Hz)</label>
                        <input type="number" class="form-control" id="sampling_rate" name="sampling_rate" value="250">
                    </div>
                    <div class="mb-3">
                        <label for="data_format" class="form-label">Data Format</label>
                        <select class="form-select" id="data_format" name="data_format">
                            <option value="samples_leads" selected>Samples × Leads</option>
                            <option value="leads_samples">Leads × Samples</option>
                        </select>
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Show MATLAB file options
 */
function showMatlabOptions() {
    const optionsContainer = document.querySelector('.file-options');
    if (optionsContainer) {
        optionsContainer.innerHTML = `
            <div class="card mt-3">
                <div class="card-header">MATLAB Options</div>
                <div class="card-body">
                    <div class="mb-3">
                        <label for="variable_name" class="form-label">Variable Name</label>
                        <input type="text" class="form-control" id="variable_name" name="variable_name" placeholder="e.g., ecg_data">
                    </div>
                    <div class="mb-3">
                        <label for="sampling_rate" class="form-label">Sampling Rate (Hz)</label>
                        <input type="number" class="form-control" id="sampling_rate" name="sampling_rate" value="250">
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Initialize analysis form
 */
function initAnalysisForm() {
    console.log('Initializing analysis form');
    
    // Toggle advanced options
    const advancedToggle = document.querySelector('.advanced-toggle');
    const advancedOptions = document.querySelector('.advanced-options');
    
    if (advancedToggle && advancedOptions) {
        advancedToggle.addEventListener('click', function(e) {
            e.preventDefault();
            advancedOptions.classList.toggle('d-none');
            
            // Update toggle text
            if (advancedOptions.classList.contains('d-none')) {
                advancedToggle.innerHTML = '<i class="fas fa-cog me-1"></i> Show Advanced Options';
            } else {
                advancedToggle.innerHTML = '<i class="fas fa-cog me-1"></i> Hide Advanced Options';
            }
        });
    }
    
    // Handle model selection
    const modelTypeSelect = document.querySelector('#model_type');
    const mlModelSelect = document.querySelector('#ml_model');
    const dlModelSelect = document.querySelector('#dl_model');
    
    if (modelTypeSelect && mlModelSelect && dlModelSelect) {
        modelTypeSelect.addEventListener('change', function() {
            if (this.value === 'ml') {
                mlModelSelect.parentElement.classList.remove('d-none');
                dlModelSelect.parentElement.classList.add('d-none');
            } else {
                mlModelSelect.parentElement.classList.add('d-none');
                dlModelSelect.parentElement.classList.remove('d-none');
            }
        });
    }
}

/**
 * Add fade-in animation to cards
 */
function animateCards() {
    const cards = document.querySelectorAll('.card');
    
    cards.forEach((card, index) => {
        // Add fade-in class with delay based on index
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
}

/**
 * Format a date string
 * 
 * @param {string} dateString - The date string to format
 * @returns {string} - The formatted date string
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Format a file size in bytes to a human-readable string
 * 
 * @param {number} bytes - The file size in bytes
 * @returns {string} - The formatted file size string
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

